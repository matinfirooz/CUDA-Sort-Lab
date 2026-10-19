#include "cudasort/sort.cuh"
#include "cudasort/cuda_utils.cuh"
#include "cudasort/input.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using cudasort::Algorithm;

namespace {

struct Arguments {
    std::string algorithm = "radix", pattern = "random", csv;
    std::size_t n = 1 << 20;
    std::uint32_t seed = 42;
    int repeats = 3, device = 0, minimum = -4096, maximum = 4095;
    bool allow_slow = false, self_test = false, list = false, help = false;
};

unsigned long long unsigned_number(const std::string& value) {
    if (value.empty() || value[0] == '-') throw std::invalid_argument("Expected unsigned number");
    std::size_t consumed = 0;
    const auto result = std::stoull(value, &consumed);
    if (consumed != value.size()) throw std::invalid_argument("Invalid number: " + value);
    return result;
}

int signed_number(const std::string& value) {
    std::size_t consumed = 0;
    const auto result = std::stoll(value, &consumed);
    if (consumed != value.size() || result < INT_MIN || result > INT_MAX)
        throw std::invalid_argument("Invalid 32-bit number: " + value);
    return static_cast<int>(result);
}

Arguments parse(int argc, char** argv) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help" || option == "-h") args.help = true;
        else if (option == "--list") args.list = true;
        else if (option == "--self-test") args.self_test = true;
        else if (option == "--allow-slow") args.allow_slow = true;
        else {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + option);
            const std::string value = argv[++i];
            if (option == "--algo") args.algorithm = value;
            else if (option == "--pattern") args.pattern = value;
            else if (option == "--csv") args.csv = value;
            else if (option == "--n") {
                const auto n = unsigned_number(value);
                if (n > cudasort::max_items) throw std::invalid_argument("n exceeds 2^26");
                args.n = static_cast<std::size_t>(n);
            } else if (option == "--seed") {
                const auto seed = unsigned_number(value);
                if (seed > std::numeric_limits<std::uint32_t>::max())
                    throw std::invalid_argument("Seed exceeds uint32 range");
                args.seed = static_cast<std::uint32_t>(seed);
            } else if (option == "--repeats") args.repeats = signed_number(value);
            else if (option == "--device") args.device = signed_number(value);
            else if (option == "--min") args.minimum = signed_number(value);
            else if (option == "--max") args.maximum = signed_number(value);
            else throw std::invalid_argument("Unknown option: " + option);
        }
    }
    if (args.repeats < 1) throw std::invalid_argument("repeats must be positive");
    if (args.minimum > args.maximum) throw std::invalid_argument("min exceeds max");
    return args;
}

void help() {
    std::cout <<
        "CUDA Sort Lab: ascending signed int32 keys\n"
        "  --list                        List 20 algorithms (no GPU needed)\n"
        "  --algo NAME|all|NAME,NAME       Default: radix\n"
        "  --n N                         Default: 1048576; maximum: 2^26\n"
        "  --pattern NAME                random|sorted|reverse|equal|few-unique|\n"
        "                                extremes|nearly-sorted|organ-pipe\n"
        "  --seed N                      Default: 42\n"
        "  --min N --max N               Random key range; defaults -4096,4095\n"
        "  --repeats N                   Timed runs after one warm-up; default: 3\n"
        "  --csv PATH                    Write measurements and skip reasons\n"
        "  --device N                    Device ordinal; default: 0\n"
        "  --allow-slow                  Override educational size guards\n"
        "  --self-test                   GPU regression suite\n"
        "  --help                        Show this message\n";
}

std::vector<Algorithm> selected_algorithms(const std::string& selection) {
    std::vector<Algorithm> result;
    if (selection == "all") {
        for (const auto& entry : cudasort::algorithms) result.push_back(entry.id);
    } else {
        std::istringstream stream(selection);
        std::string name;
        while (std::getline(stream, name, ',')) result.push_back(cudasort::parse_algorithm(name));
        if (result.empty() || selection.back() == ',')
            throw std::invalid_argument("Empty algorithm selection");
    }
    return result;
}

cudasort::Options options_for(const std::vector<int>& input) {
    cudasort::Options options;
    if (!input.empty()) {
        const auto mm = std::minmax_element(input.begin(), input.end());
        options.minimum = *mm.first;
        options.maximum = *mm.second;
    }
    return options;
}

bool counting_fits(const cudasort::Options& options) {
    const auto bins = static_cast<long long>(options.maximum) - options.minimum + 1;
    return bins > 0 && static_cast<unsigned long long>(bins) <= options.max_counting_bins;
}

void verify(const std::vector<int>& actual, const std::vector<int>& expected,
            std::string_view algorithm) {
    if (actual != expected) {
        const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
        throw std::runtime_error(std::string(algorithm) + " mismatches std::sort at index " +
                                 std::to_string(mismatch.first - actual.begin()));
    }
}

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct Timing { double sort_ms, total_ms; };

Timing run_once(Algorithm algorithm, const std::vector<int>& input,
                 const std::vector<int>& expected, int* device,
                 const cudasort::Options& options, cudaStream_t stream) {
    std::vector<int> actual(input.size());
    const auto bytes = input.size() * sizeof(int);
    const auto start = Clock::now();
    if (bytes) CS_CUDA(cudaMemcpyAsync(device, input.data(), bytes, cudaMemcpyHostToDevice, stream));
    CS_CUDA(cudaStreamSynchronize(stream));
    const auto sort_start = Clock::now();
    cudasort::sort(device, input.size(), algorithm, options, stream);
    const auto sort_end = Clock::now();
    if (bytes) CS_CUDA(cudaMemcpyAsync(actual.data(), device, bytes, cudaMemcpyDeviceToHost, stream));
    CS_CUDA(cudaStreamSynchronize(stream));
    const auto end = Clock::now();
    verify(actual, expected, cudasort::info(algorithm).name);
    return {milliseconds(sort_start, sort_end), milliseconds(start, end)};
}

template<class Function>
void expect_invalid(Function function, const char* test) {
    bool rejected = false;
    try { function(); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error(std::string("Missing rejection: ") + test);
}

void self_test(cudaStream_t stream) {
    const std::size_t sizes[]{0, 1, 2, 3, 7, 31, 32, 33, 255, 256, 257,
                              511, 512, 513, 1023, 1024, 1025};
    std::size_t checked = 0, skipped = 0;
    auto check_case = [&](Algorithm algorithm, const std::vector<int>& input) {
        auto options = options_for(input);
        if (algorithm == Algorithm::Counting && !counting_fits(options)) { ++skipped; return; }
        auto expected = input;
        std::sort(expected.begin(), expected.end());
        cudasort::DeviceBuffer<int> device(input.size());
        run_once(algorithm, input, expected, device.get(), options, stream);
        ++checked;
    };
    for (const auto& entry : cudasort::algorithms) {
        for (auto n : sizes)
            for (auto pattern : cudasort::patterns)
                check_case(entry.id, cudasort::make_input(n, pattern, 17, -17, 19));
        std::cout << "PASS small cases: " << entry.name << '\n';
    }
    const Algorithm large[]{Algorithm::Bitonic, Algorithm::Merge, Algorithm::Radix,
                             Algorithm::Counting, Algorithm::Quick, Algorithm::Thrust, Algorithm::Cub};
    for (auto n : {65537u, 262145u})
        for (auto pattern : {"random", "few-unique", "equal", "extremes"})
            for (auto algorithm : large)
                check_case(algorithm, cudasort::make_input(n, pattern, 91, -4096, 4095));

    cudasort::DeviceBuffer<int> single(1);
    const int value = 8;
    CS_CUDA(cudaMemcpyAsync(single.get(), &value, sizeof(int), cudaMemcpyHostToDevice, stream));
    cudasort::Options options;
    options.minimum = 0;
    options.maximum = 7;
    expect_invalid([&] { cudasort::sort(single.get(), 1, Algorithm::Counting, options, stream); },
                   "counting out-of-range singleton");
    options.minimum = INT_MIN;
    options.maximum = INT_MAX;
    expect_invalid([&] { cudasort::sort(single.get(), 1, Algorithm::Counting, options, stream); },
                   "counting oversized range");
    expect_invalid([&] { cudasort::sort(nullptr, 1, Algorithm::Radix, {}, stream); }, "null pointer");
    expect_invalid([&] { cudasort::sort(nullptr, cudasort::max_items + 1, Algorithm::Radix, {}, stream); },
                   "oversized input");
    expect_invalid([&] { cudasort::sort(nullptr, 0, static_cast<Algorithm>(999), {}, stream); },
                   "invalid algorithm");
    std::cout << "GPU tests passed: " << checked << "; counting range skips: " << skipped
              << "; invalid-input checks: 5\n";
}

std::string csv_quote(std::string_view text) {
    std::string out = "\"";
    for (char c : text) { if (c == '\"') out += '\"'; out += c; }
    return out + "\"";
}

int benchmark(const Arguments& args, const std::vector<Algorithm>& selected,
              const std::vector<int>& input, cudaStream_t stream) {
    auto expected = input;
    const auto cpu_start = Clock::now();
    std::sort(expected.begin(), expected.end());
    const double cpu_ms = milliseconds(cpu_start, Clock::now());
    const auto options = options_for(input);
    std::ofstream csv;
    if (!args.csv.empty()) {
        csv.open(args.csv);
        if (!csv) throw std::runtime_error("Cannot open CSV: " + args.csv);
        csv << "algorithm,n,pattern,seed,repeats,status,sort_ms,end_to_end_ms,cpu_sort_ms,million_keys_s,reason\n";
        csv << std::setprecision(9);
    }
    std::cout << std::left << std::setw(12) << "algorithm" << std::setw(8) << "status"
              << std::right << std::setw(14) << "sort ms" << std::setw(16) << "end-to-end ms"
              << std::setw(14) << "M keys/s" << '\n';
    std::size_t passed = 0;
    for (auto algorithm : selected) {
        const auto& info = cudasort::info(algorithm);
        std::string reason;
        if (!args.allow_slow && info.suggested_limit && args.n > info.suggested_limit)
            reason = "educational size guard; use --allow-slow";
        if (algorithm == Algorithm::Counting && !counting_fits(options))
            reason = "observed key range exceeds 2^20 counting bins";
        std::cout << std::left << std::setw(12) << info.name;
        if (!reason.empty()) {
            std::cout << "SKIP    " << reason << '\n';
            if (csv) csv << info.name << ',' << args.n << ',' << args.pattern << ',' << args.seed
                         << ',' << args.repeats << ",SKIP,,," << cpu_ms << ",," << csv_quote(reason) << '\n';
            continue;
        }
        cudasort::DeviceBuffer<int> device(input.size());
        run_once(algorithm, input, expected, device.get(), options, stream);  // Verified warm-up.
        double sort_ms = 0, total_ms = 0;
        for (int repeat = 0; repeat < args.repeats; ++repeat) {
            const auto timing = run_once(algorithm, input, expected, device.get(), options, stream);
            sort_ms += timing.sort_ms;
            total_ms += timing.total_ms;
        }
        sort_ms /= args.repeats;
        total_ms /= args.repeats;
        const double throughput = sort_ms > 0 ? static_cast<double>(args.n) / (1000.0 * sort_ms) : 0;
        std::cout << std::setw(8) << "PASS" << std::right << std::fixed << std::setprecision(4)
                  << std::setw(14) << sort_ms << std::setw(16) << total_ms
                  << std::setw(14) << throughput << '\n';
        if (csv) csv << info.name << ',' << args.n << ',' << args.pattern << ',' << args.seed
                     << ',' << args.repeats << ",PASS," << sort_ms << ',' << total_ms << ','
                     << cpu_ms << ',' << throughput << ",\n";
        ++passed;
    }
    if (csv.is_open()) {
        csv.flush();
        if (!csv) throw std::runtime_error("Writing CSV failed");
    }
    std::cout << "CPU std::sort: " << cpu_ms << " ms; " << passed << " algorithm(s) verified.\n";
    return passed ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse(argc, argv);
        if (args.help) { help(); return 0; }
        if (args.list) {
            for (const auto& entry : cudasort::algorithms)
                std::cout << std::left << std::setw(12) << entry.name << entry.execution << '\n';
            return 0;
        }
        // Reject malformed requests before probing the CUDA driver.
        const auto selected = selected_algorithms(args.algorithm);
        if (std::find(cudasort::patterns.begin(), cudasort::patterns.end(), args.pattern) ==
            cudasort::patterns.end()) throw std::invalid_argument("Unknown input pattern");
        int devices = 0;
        CS_CUDA(cudaGetDeviceCount(&devices));
        if (args.device < 0 || args.device >= devices)
            throw std::runtime_error("Requested CUDA device is unavailable");
        CS_CUDA(cudaSetDevice(args.device));
        cudaDeviceProp properties{};
        CS_CUDA(cudaGetDeviceProperties(&properties, args.device));
        int driver = 0, runtime = 0;
        CS_CUDA(cudaDriverGetVersion(&driver));
        CS_CUDA(cudaRuntimeGetVersion(&runtime));
        std::cerr << "GPU: " << properties.name << "; SM " << properties.major << properties.minor
                  << "; driver " << driver << "; runtime " << runtime << '\n';
        cudasort::Stream stream;
        if (args.self_test) { self_test(stream); return 0; }
        const auto input = cudasort::make_input(args.n, args.pattern, args.seed, args.minimum, args.maximum);
        return benchmark(args, selected, input, stream);
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
