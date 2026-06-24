/*
 * Copyright (c) 2024, Alibaba Group;
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * ---
 * DECOUPLED REPLAY: Complete flow file parser.
 * Parses layer metadata header + 16-field flow body.
 * Based on: loadFlowsFromFile() in MockNcclGroup.cc:2245-2265
 *
 * Flow file format:
 *   [line 1]    total_flows layer_count
 *   [line 2..L] layer: <N>  total_flows: <F>  compute_before_ns: <C>
 *   [body]      flow_id src dest flow_size channel_id chunk_id chunk_count conn_type
 *               start_time pg maxPacketCount port dport
 *               np prev[0..np-1]
 *               layer_num group_type op loopstate
 */

#ifndef __DECOUPLED_FLOW_READER_H__
#define __DECOUPLED_FLOW_READER_H__

#include "common_types.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <iostream>

// ============================================================================
// FlowFileRecord: parsed flow body record (16 fields)
// ============================================================================

struct FlowFileRecord {
    uint32_t flow_id = 0;
    uint32_t src = 0;
    uint32_t dst = 0;
    uint64_t flow_size = 0;
    int channel_id = 0;
    int chunk_id = 0;
    int chunk_count = 0;
    std::string conn_type;
    double start_time = 0.0;
    uint32_t pg = 0;
    uint32_t maxPacketCount = 0;
    uint32_t port = 0;
    uint32_t dport = 0;
    std::vector<uint32_t> prev;
    uint32_t layer_num = 0;
    uint32_t group_type = 0;
    uint32_t op = 0;

    bool valid() const {
    bool valid() const {
        return flow_size > 0 || (src != dst);
    }
};

// ============================================================================
// LayerMeta: per-layer compute timing metadata
// ============================================================================

struct LayerMeta {
    uint32_t layer_num = 0;
    uint32_t total_flows = 0;
    uint64_t compute_before_ns = 0;  // GPU compute time before this layer's communication starts
};

using LayerMetaMap = std::map<uint32_t, LayerMeta>;  // layer_num → LayerMeta


// ============================================================================
// LoadFlows: Parse complete flow file
// Extracted from: MockNcclGroup.cc:2245-2265 (loadFlowsFromFile)
// ============================================================================

// Extracted from: MockNcclGroup.cc:2245-2265 (parse loop pattern)
inline std::vector<FlowFileRecord> LoadFlows(const std::string& flow_file_path,
                                              LayerMetaMap& layer_meta_out) {
    std::vector<FlowFileRecord> flows;
    layer_meta_out.clear();

    std::ifstream ff(flow_file_path);
    if (!ff.is_open()) {
        std::cerr << "[LoadFlows] ERROR: Cannot open flow file: "
                  << flow_file_path << std::endl;
        return flows;
    }

    // ── Parse header: total_flows [layer_count] ──
    std::string header_line;
    if (!std::getline(ff, header_line) || header_line.empty()) {
        std::cerr << "[LoadFlows] ERROR: Empty or invalid flow file: "
                  << flow_file_path << std::endl;
        return flows;
    }

    std::istringstream hs(header_line);
    uint32_t total = 0;
    uint32_t layer_count = 0;
    hs >> total;
    // layer_count is optional (old format: just "total")
    if (!(hs >> layer_count)) {
        layer_count = 0;
    }

    if (total == 0) {
        std::cerr << "[LoadFlows] WARNING: Flow file header says 0 flows." << std::endl;
        return flows;
    }

    flows.reserve(total);

    // ── Parse layer metadata (if layer_count > 0) ──
    for (uint32_t l = 0; l < layer_count; l++) {
        std::string meta_line;
        if (!std::getline(ff, meta_line)) {
            std::cerr << "[LoadFlows] ERROR: Expected layer metadata line "
                      << l << " but got EOF" << std::endl;
            break;
        }
        // Format: "layer: <N>  total_flows: <F>  compute_before_ns: <C>"
        LayerMeta lm;
        std::string kw_layer, kw_flows, kw_compute;
        std::istringstream ms(meta_line);
        ms >> kw_layer >> lm.layer_num;               // "layer:" N
        ms >> kw_flows >> lm.total_flows;             // "total_flows:" F
        ms >> kw_compute >> lm.compute_before_ns;     // "compute_before_ns:" C
        if (!ms.fail() &&
            kw_layer == "layer:" &&
            kw_flows == "total_flows:" &&
            kw_compute == "compute_before_ns:") {
            layer_meta_out[lm.layer_num] = lm;
        } else {
            std::cerr << "[LoadFlows] WARNING: Bad layer metadata line, skipping: "
                      << meta_line << std::endl;
        }
    }

    // ── Parse flow body ──
    uint32_t body_line_num = 1 + layer_count;  // header + metadata lines
    std::string line;
    while (std::getline(ff, line)) {
        body_line_num++;
        if (line.empty()) continue;

        std::istringstream is(line);
        FlowFileRecord r;

        // Fields 1-8
        if (!(is >> r.flow_id >> r.src >> r.dst >> r.flow_size
                  >> r.channel_id >> r.chunk_id >> r.chunk_count
                  >> r.conn_type)) {
            std::cerr << "[LoadFlows] ERROR: Line " << body_line_num
                      << " truncated (cannot read fields 1-8)" << std::endl;
            continue;
        }

        // Fields 9-14: start_time, pg, maxPacketCount, port, dport, np
        double st; uint32_t pg, mpc, port, dport; uint32_t np;
        if (!(is >> st >> pg >> mpc >> port >> dport >> np)) {
            std::cerr << "[LoadFlows] ERROR: Line " << body_line_num
                      << " truncated (cannot read fields 9-14)" << std::endl;
            continue;
        }
        r.start_time = st;
        r.pg = pg;
        r.maxPacketCount = mpc;
        r.port = port;
        r.dport = dport;

        // Field prev[] (variable length)
        r.prev.reserve(np);
        for (uint32_t j = 0; j < np; j++) {
            uint32_t pid;
            if (!(is >> pid)) {
                std::cerr << "[LoadFlows] ERROR: Line " << body_line_num
                          << " truncated (prev[" << j << "/" << np << "])" << std::endl;
                break;
            }
            r.prev.push_back(pid);
        }

        // Fields: layer_num, group_type, op, loopstate (16 fields total)
        if (!(is >> r.layer_num >> r.group_type >> r.op >> r.loopstate)) {
            std::cerr << "[LoadFlows] ERROR: Line " << body_line_num
                      << " truncated (cannot read layer_num/group_type/op/loopstate)"
                      << std::endl;
            continue;
        }

        flows.push_back(r);
    }

    ff.close();

    std::cout << "[LoadFlows] Parsed " << flows.size() << " flows, "
              << layer_meta_out.size() << " layer metadata entries from "
              << flow_file_path << " (header said " << total << ")" << std::endl;

    return flows;
}

#endif // __DECOUPLED_FLOW_READER_H__
