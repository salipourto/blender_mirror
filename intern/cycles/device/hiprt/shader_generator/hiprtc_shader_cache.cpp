//SPDX - License - Identifier : GPL - 2.0 - or -later


#include <fstream>
#include <vector>
#include "hiprt.h"
#include "include/hipew.h"

# include "device/kernel.h"
#include "device/hiprt/rtc_util.h"

//HIPRTSDK should be added to path

int main(int argc, const char *argv[])
{
  std::string source_path;
  std::string include_path;
  std::string include_nano;
  std::string out_path;

  for (int i = 1; i < argc; i++) {
    std::string current_arg = (argv[i]);

    if (current_arg == "--include_path") {

      include_path = argv[i + 1];
    }
    if (current_arg == "--nano_include") {

      include_nano = "-I" + std::string(argv[i + 1]);
    }
    else if (current_arg == "--source_path") {

      source_path = argv[i + 1];
    }
    else if (current_arg == "--output_path") {
      out_path = argv[i + 1];
    }
  }

  hiprtContextCreationInput hiprt_context_input = {0};
  hiprt_context_input.ctxt = 0;
  hiprt_context_input.device = 0;
  hiprt_context_input.deviceType = hiprtDeviceAMD;

  hiprtContext hiprt_context;

  hiprtError rt_result = hiprtCreateContext(
      HIPRT_API_VERSION, hiprt_context_input, &hiprt_context);

  std::vector<const char *> kernel_function_names;
  std::vector<std::string> function_name_str;
  hiprt_rtc_helper::get_kernel_names(function_name_str, kernel_function_names);

  std::vector<hiprtFuncNameSet> func_name_sets;
  func_name_sets.resize(hiprt_rtc_helper::Max_Intersect_Filter_Function *
                        hiprt_rtc_helper::Max_Primitive_Type);

  hiprt_rtc_helper::get_custom_function_names(func_name_sets.data());

  bool use_shared_stack = false;
#ifdef HIPRT_SHARED_STACK
  use_shared_stack = true;
#endif

  std::vector<const char *> rtc_options;
  std::string include = "-I" + include_path;
  rtc_options.push_back(include.c_str());
  hiprt_rtc_helper::shared_stack_property stack_proeprty;
  hiprt_rtc_helper::get_compiler_options(rtc_options, stack_proeprty, use_shared_stack);
#if 0  // def WITH_NANOVDB
   rtc_options.push_back("-DWITH_NANOVDB");
   rtc_options.push_back(include_nano.c_str());
   //std::string include_nanovdb =
#endif

#ifdef WITH_CYCLES_DEBUG
  rtc_options.push_back("-DWITH_CYCLES_DEBUG");
#endif

  std::ifstream source_file(source_path);
  std::string src_txt((std::istreambuf_iterator<char>(source_file)),
                      (std::istreambuf_iterator<char>()));

  source_file.close();

  // compile for different gpu

  const char *gpu_targets[] = {"gfx900",
                               "gfx906",
                               "gfx1010",
                               "gfx1011",
                               "gfx1012",
                               "gfx1013",
                               "gfx1100",
                               "gfx1102",
                               "gfx1030",
                               "gfx1031",
                               "gfx1032",
                               "gfx1033",
                               "gfx1034",
                               "gfx1035",
                               "gfx1036"};

  int num_targets = sizeof(gpu_targets) / sizeof(char *);
  //add code to check whether recompile is needed

  int gpu_opt_index = rtc_options.size();
  rtc_options.resize(gpu_opt_index + 1);

  for (int i = 0; i < num_targets; i++) {

    hiprtcProgram intersection = 0;
    std::vector<char> intersection_binary;

    std::string current_gpu_target = "--gpu-architecture=" + std::string(gpu_targets[i]);
    rtc_options[gpu_opt_index] = current_gpu_target.c_str();

    const std::string output_name = out_path + "//" + "kernel_rt_" + std::string(gpu_targets[i]) +
                                    ".fatbin";

    hiprtError e = hiprtBuildTraceProgram(hiprt_context,
                                          kernel_function_names.size(),
                                          kernel_function_names.data(),
                                          src_txt.c_str(),  // source code
                                          0,                // program name, can be null
                                          0,
                                          0,
                                          0,
                                          rtc_options.size(),
                                          rtc_options.data(),
                                          hiprt_rtc_helper::Max_Primitive_Type,
                                          hiprt_rtc_helper::Max_Intersect_Filter_Function,
                                          func_name_sets.data(),
                                          &intersection);

    size_t binary_size = 0;
    if (e == 0)
      e = hiprtBuildTraceGetBinary(&intersection, &binary_size, nullptr);
    else
      printf("Failed to compile for %s\n", current_gpu_target.c_str());

    if (binary_size > 0) {
      intersection_binary.resize(binary_size);
      e = hiprtBuildTraceGetBinary(&intersection, &binary_size, intersection_binary.data());

      std::ofstream shader_binary;
      shader_binary.open(output_name.c_str(), std::ios::binary);
      shader_binary.write(intersection_binary.data(), binary_size);
      shader_binary.close();
    }
  }

  return 0;
}
