# Trimmed from upstream tevador/RandomX CMakeLists.txt (commit 1e9d4b2).
# Builds only the `randomx` static library target; upstream test/codegen/benchmark
# executables and install() rules are removed. Arch detection (x86_64 / ARMv8 /
# PowerPC / RISC-V) is kept as upstream ships it.

# The JIT static stubs are GNU-syntax .S files (assembler-with-cpp). The parent project
# only enables C/CXX, so enable ASM here and tag those sources LANGUAGE ASM -- compiling
# them as C (upstream's ccache "cheat") makes the C preprocessor choke on the `#` asm
# comment lines. clang's ASM driver runs the asm preprocessor and handles them.
enable_language(ASM)

set(randomx_sources
src/aes_hash.cpp
src/argon2_ref.c
src/argon2_ssse3.c
src/argon2_avx2.c
src/bytecode_machine.cpp
src/cpu.cpp
src/dataset.cpp
src/soft_aes.cpp
src/virtual_memory.c
src/vm_interpreted.cpp
src/allocator.cpp
src/assembly_generator_x86.cpp
src/instruction.cpp
src/randomx.cpp
src/superscalar.cpp
src/vm_compiled.cpp
src/vm_interpreted_light.cpp
src/argon2_core.c
src/blake2_generator.cpp
src/instructions_portable.cpp
src/reciprocal.c
src/virtual_machine.cpp
src/vm_compiled_light.cpp
src/blake2/blake2b.c)

if(NOT ARCH_ID)
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "")
    set(CMAKE_SYSTEM_PROCESSOR ${CMAKE_HOST_SYSTEM_PROCESSOR})
  endif()
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ARCH_ID)
endif()

if(NOT ARM_ID)
  set(ARM_ID "${ARCH_ID}")
endif()

if(NOT ARCH)
  set(ARCH "default")
endif()

include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)

function(add_flag flag)
  string(REPLACE "-" "_" supported_cxx ${flag}_cxx)
  check_cxx_compiler_flag(${flag} ${supported_cxx})
  if(${${supported_cxx}})
    message(STATUS "Setting CXX flag ${flag}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
  endif()
  string(REPLACE "-" "_" supported_c ${flag}_c)
  check_c_compiler_flag(${flag} ${supported_c})
  if(${${supported_c}})
    message(STATUS "Setting C flag ${flag}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}" PARENT_SCOPE)
  endif()
endfunction()

# x86-64
if ((CMAKE_SIZEOF_VOID_P EQUAL 8) AND (ARCH_ID STREQUAL "x86_64" OR ARCH_ID STREQUAL "x86-64" OR ARCH_ID STREQUAL "amd64"))
  list(APPEND randomx_sources
    src/jit_compiler_x86.cpp)

  if(MSVC)
    enable_language(ASM_MASM)
    list(APPEND randomx_sources src/jit_compiler_x86_static.asm)
    set_property(SOURCE src/jit_compiler_x86_static.asm PROPERTY LANGUAGE ASM_MASM)
    set_source_files_properties(src/argon2_avx2.c COMPILE_FLAGS /arch:AVX2)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/asm/configuration.asm
      COMMAND powershell -ExecutionPolicy Bypass -File h2inc.ps1 ..\\src\\configuration.h > ..\\src\\asm\\configuration.asm SET ERRORLEVEL = 0
      COMMENT "Generating configuration.asm at ${CMAKE_CURRENT_SOURCE_DIR}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vcxproj)
    add_custom_target(generate-asm
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/asm/configuration.asm)
  else()
    list(APPEND randomx_sources src/jit_compiler_x86_static.S)
    set_property(SOURCE src/jit_compiler_x86_static.S PROPERTY LANGUAGE ASM)
    set_property(SOURCE src/jit_compiler_x86_static.S PROPERTY XCODE_EXPLICIT_FILE_TYPE sourcecode.asm)
    if(ARCH STREQUAL "native")
      add_flag("-march=native")
    else()
      add_flag("-maes")
      check_c_compiler_flag(-mssse3 HAVE_SSSE3)
      if(HAVE_SSSE3)
        set_source_files_properties(src/argon2_ssse3.c COMPILE_FLAGS -mssse3)
      endif()
      check_c_compiler_flag(-mavx2 HAVE_AVX2)
      if(HAVE_AVX2)
        set_source_files_properties(src/argon2_avx2.c COMPILE_FLAGS -mavx2)
      endif()
    endif()
  endif()
endif()

# PowerPC
if(ARCH_ID STREQUAL "ppc64" OR ARCH_ID STREQUAL "ppc64le")
  if(ARCH STREQUAL "native")
    add_flag("-mcpu=native")
  endif()
endif()

# ARMv8
if(ARM_ID STREQUAL "aarch64" OR ARM_ID STREQUAL "arm64" OR ARM_ID STREQUAL "armv8-a")
  list(APPEND randomx_sources
    src/jit_compiler_a64_static.S
    src/jit_compiler_a64.cpp)
  set_property(SOURCE src/jit_compiler_a64_static.S PROPERTY LANGUAGE ASM)
  set_property(SOURCE src/jit_compiler_a64_static.S PROPERTY XCODE_EXPLICIT_FILE_TYPE sourcecode.asm)

  include(CheckIncludeFile)
  check_include_file(asm/hwcap.h HAVE_HWCAP)
  if(HAVE_HWCAP)
    add_definitions(-DHAVE_HWCAP)
  endif()

  if(ARCH STREQUAL "native")
    add_flag("-march=native")
  else()
    add_flag("-march=armv8-a+crypto")
  endif()
endif()

add_library(randomx STATIC ${randomx_sources})

if(TARGET generate-asm)
  add_dependencies(randomx generate-asm)
endif()

set_property(TARGET randomx PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET randomx PROPERTY CXX_STANDARD 11)
set_property(TARGET randomx PROPERTY CXX_STANDARD_REQUIRED ON)
target_include_directories(randomx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
