#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Caps {
    int ctx_tokens;
    int batch;
    int ubatch;
    int parallel;
    int ngl;
    int cgroup_memory_max_mb;
    int vram_budget_mb;
};

struct Lane {
    std::string id;
    std::string model_role;
    std::string backend;
    std::string slot;
    std::string binary;
    std::string model_path;
    int port;
    bool current_artifact;
    bool switch_exclusive;
    Caps caps;
};

static std::string q(const std::string &s) {
    std::string out = "\"";
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    out += "\"";
    return out;
}

static std::vector<Lane> lanes() {
    const std::string llama = "01_REPOS/llama.cpp/build-cuda/bin/llama-server";
    const std::string prism = "01_REPOS/prismml_llama.cpp/build-cuda/bin/llama-server";
    const std::string mamba = "03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/Falcon3-Mamba-7B-Instruct-Q2_K.gguf";
    const std::string bonsai = "03_VAULT/models/prism-ml/Ternary-Bonsai-4B-gguf/Ternary-Bonsai-4B-Q2_0.gguf";
    const std::string deepseek = "03_VAULT/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf";
    return {
        {"mamba7b_ram", "Mamba 7B ternary/low-bit RAM always-on listener", "llama_cpp_gguf", "ram_always", llama, mamba, 8081, true, false, {512, 32, 8, 1, 0, 4096, 0}},
        {"bonsai4b_ram_a", "Bonsai 4B ternary RAM always-on worker A", "prismml_llama_cpp_gguf", "ram_always", prism, bonsai, 8082, true, false, {512, 64, 16, 1, 0, 2048, 0}},
        {"bonsai4b_ram_b", "Bonsai 4B ternary RAM always-on worker B", "prismml_llama_cpp_gguf", "ram_always", prism, bonsai, 8084, true, false, {512, 64, 16, 1, 0, 2048, 0}},
        {"mamba7b_vram_always", "Mamba 7B ternary/low-bit VRAM always-on stateless lane", "llama_cpp_gguf", "vram_always", llama, mamba, 8083, true, false, {128, 16, 4, 1, 24, 4096, 1400}},
        {"bonsai4b_vram_switch", "Bonsai 4B ternary switchable VRAM lane", "prismml_llama_cpp_gguf", "vram_switchable", prism, bonsai, 8085, true, true, {512, 64, 16, 1, 99, 2048, 1800}},
        {"deepseek_r1_int8_switch", "DeepSeek R1 distilled switchable GPU reasoning lane", "llama_cpp_gguf", "vram_switchable", llama, deepseek, 8080, true, true, {1024, 128, 32, 1, 99, 3072, 1800}},
    };
}

static const Lane *find_lane(const std::string &id) {
    static const std::vector<Lane> ls = lanes();
    for (const auto &lane : ls) {
        if (lane.id == id) return &lane;
    }
    return nullptr;
}

static std::vector<std::string> argv_for(const Lane &lane) {
    return {
        lane.binary,
        "-m", lane.model_path,
        "--host", "127.0.0.1",
        "--port", std::to_string(lane.port),
        "-ngl", std::to_string(lane.caps.ngl),
        "-c", std::to_string(lane.caps.ctx_tokens),
        "--parallel", std::to_string(lane.caps.parallel),
        "--batch-size", std::to_string(lane.caps.batch),
        "--ubatch-size", std::to_string(lane.caps.ubatch),
        "--cache-ram", "0",
        "--no-warmup"
    };
}

static void print_string_array(std::ostream &os, const std::vector<std::string> &items) {
    os << "[";
    for (size_t i = 0; i < items.size(); ++i) {
        if (i) os << ",";
        os << q(items[i]);
    }
    os << "]";
}

static void print_caps(std::ostream &os, const Caps &caps) {
    os << "{"
       << q("kv_ctx_tokens") << ":" << caps.ctx_tokens << ","
       << q("batch") << ":" << caps.batch << ","
       << q("ubatch") << ":" << caps.ubatch << ","
       << q("parallel") << ":" << caps.parallel << ","
       << q("ngl") << ":" << caps.ngl << ","
       << q("cgroup_memory_max_mb") << ":" << caps.cgroup_memory_max_mb << ","
       << q("vram_budget_mb") << ":" << caps.vram_budget_mb
       << "}";
}

static bool caps_valid(const Caps &caps, std::string &err) {
    if (caps.ctx_tokens <= 0) { err = "invalid_ctx"; return false; }
    if (caps.parallel != 1) { err = "invalid_parallel"; return false; }
    if (caps.batch <= 0 || caps.ubatch <= 0 || caps.ubatch > caps.batch) { err = "invalid_batch"; return false; }
    if (caps.cgroup_memory_max_mb <= 0) { err = "missing_cgroup_memory_max"; return false; }
    if (caps.ngl < 0) { err = "invalid_ngl"; return false; }
    return true;
}

static int print_plan() {
    const auto ls = lanes();
    std::cout << "{"
              << q("schema") << ":" << q("lucidota.ternary_harness.plan.v1") << ","
              << q("source") << ":" << q("TOOLS/RUNTIME/ternary_harness/lucidota_ternary_harness.cpp") << ","
              << q("truth_note") << ":" << q("Harness controls current GGUF artifacts and can admit BitNet-native backends; current Falcon3-Mamba local artifact is GGUF Q2_K, not proven BitNet-native.") << ","
              << q("slot_policy") << ":{"
                  << q("ram_always") << ":{" << q("exclusive") << ":false},"
                  << q("vram_always") << ":{" << q("exclusive") << ":false},"
                  << q("vram_switchable") << ":{" << q("exclusive") << ":true," << q("protocol") << ":" << q("drain-stop-verify-start-health-receipt") << "}"
              << "},"
              << q("lanes") << ":[";
    for (size_t i = 0; i < ls.size(); ++i) {
        const auto &lane = ls[i];
        if (i) std::cout << ",";
        std::cout << "{"
                  << q("id") << ":" << q(lane.id) << ","
                  << q("model_role") << ":" << q(lane.model_role) << ","
                  << q("backend") << ":" << q(lane.backend) << ","
                  << q("slot") << ":" << q(lane.slot) << ","
                  << q("binary") << ":" << q(lane.binary) << ","
                  << q("model_path") << ":" << q(lane.model_path) << ","
                  << q("port") << ":" << lane.port << ","
                  << q("current_artifact") << ":" << (lane.current_artifact ? "true" : "false") << ","
                  << q("switch_exclusive") << ":" << (lane.switch_exclusive ? "true" : "false") << ","
                  << q("caps") << ":";
        print_caps(std::cout, lane.caps);
        std::cout << "}";
    }
    std::cout << "]}" << std::endl;
    return 0;
}

static int lane_argv(const std::string &id) {
    const Lane *lane = find_lane(id);
    if (!lane) {
        std::cerr << "unknown_lane: " << id << std::endl;
        return 2;
    }
    std::string err;
    if (!caps_valid(lane->caps, err)) {
        std::cerr << err << ": " << id << std::endl;
        return 3;
    }
    std::cout << "{"
              << q("schema") << ":" << q("lucidota.ternary_harness.argv.v1") << ","
              << q("lane") << ":" << q(lane->id) << ","
              << q("slot") << ":" << q(lane->slot) << ","
              << q("backend") << ":" << q(lane->backend) << ","
              << q("argv") << ":";
    print_string_array(std::cout, argv_for(*lane));
    std::cout << "," << q("caps") << ":";
    print_caps(std::cout, lane->caps);
    std::cout << "}" << std::endl;
    return 0;
}

static int check_caps(int ctx_override) {
    Caps caps{ctx_override, 16, 4, 1, 0, 1024, 0};
    std::string err;
    if (!caps_valid(caps, err)) {
        std::cerr << err << std::endl;
        return 4;
    }
    std::cout << "{" << q("schema") << ":" << q("lucidota.ternary_harness.caps_check.v1") << ","
              << q("hard_caps") << ":true," << q("caps") << ":";
    print_caps(std::cout, caps);
    std::cout << "}" << std::endl;
    return 0;
}

static int swap_plan(const std::string &id) {
    const Lane *lane = find_lane(id);
    if (!lane) {
        std::cerr << "unknown_lane: " << id << std::endl;
        return 2;
    }
    if (lane->slot != "vram_switchable") {
        std::cerr << "not_switchable_lane: " << id << std::endl;
        return 5;
    }
    const std::vector<std::string> actions = {
        "reject_new_requests",
        "drain_active_requests",
        "stop_current_switchable_pid",
        "verify_pid_dead",
        "verify_port_dead",
        "verify_vram_released",
        "start_target_lane_under_caps",
        "health_check_target_lane",
        "write_swap_receipt",
    };
    std::cout << "{" << q("schema") << ":" << q("lucidota.ternary_harness.swap_plan.v1") << ","
              << q("target_lane") << ":" << q(id) << ","
              << q("slot") << ":" << q(lane->slot) << ","
              << q("steps") << ":[";
    for (size_t i = 0; i < actions.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << "{" << q("ordinal") << ":" << (i + 1) << "," << q("action") << ":" << q(actions[i]) << "}";
    }
    std::cout << "]}" << std::endl;
    return 0;
}

static int probe_http(const std::string &id) {
    const Lane *lane = find_lane(id);
    if (!lane) {
        std::cerr << "unknown_lane: " << id << std::endl;
        return 2;
    }
    std::ostringstream cmd;
    cmd << "curl -fsS --max-time 2 http://127.0.0.1:" << lane->port << "/health >/dev/null";
    int rc = std::system(cmd.str().c_str());
    std::cout << "{" << q("schema") << ":" << q("lucidota.ternary_harness.http_probe.v1") << ","
              << q("lane") << ":" << q(id) << ","
              << q("port") << ":" << lane->port << ","
              << q("healthy") << ":" << (rc == 0 ? "true" : "false") << "}" << std::endl;
    return rc == 0 ? 0 : 6;
}

static int smoke_completion(const std::string &id) {
    const Lane *lane = find_lane(id);
    if (!lane) {
        std::cerr << "unknown_lane: " << id << std::endl;
        return 2;
    }
    std::ostringstream cmd;
    cmd << "curl -fsS --max-time 20 -H 'Content-Type: application/json' "
        << "-d '{\"prompt\":\"ALIVE?\",\"n_predict\":4,\"temperature\":0,\"cache_prompt\":false}' "
        << "http://127.0.0.1:" << lane->port << "/completion >/tmp/lucidota_ternary_harness_smoke.out";
    const int rc = std::system(cmd.str().c_str());
    const bool ok = (rc == 0);
    std::cout << "{" << q("schema") << ":" << q("lucidota.ternary_harness.smoke_completion.v1") << ","
              << q("lane") << ":" << q(id) << ","
              << q("port") << ":" << lane->port << ","
              << q("real_inference_performed") << ":" << (ok ? "true" : "false") << ","
              << q("output_capture") << ":" << q("/tmp/lucidota_ternary_harness_smoke.out") << "}" << std::endl;
    return ok ? 0 : 7;
}

static void usage() {
    std::cerr << "usage: lucidota_ternary_harness --print-plan | --lane-argv LANE | --check-caps [--ctx N] | --swap-plan LANE | --probe-http LANE | --smoke-completion LANE\n";
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args[0] == "--print-plan") return print_plan();
    if (args[0] == "--lane-argv" && args.size() >= 2) return lane_argv(args[1]);
    if (args[0] == "--swap-plan" && args.size() >= 2) return swap_plan(args[1]);
    if (args[0] == "--probe-http" && args.size() >= 2) return probe_http(args[1]);
    if (args[0] == "--smoke-completion" && args.size() >= 2) return smoke_completion(args[1]);
    if (args[0] == "--check-caps") {
        int ctx = 128;
        for (size_t i = 1; i + 1 < args.size(); ++i) {
            if (args[i] == "--ctx") ctx = std::stoi(args[i + 1]);
        }
        return check_caps(ctx);
    }
    usage();
    return 1;
}
