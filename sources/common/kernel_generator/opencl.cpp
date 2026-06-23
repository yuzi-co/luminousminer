#if defined(AMD_ENABLE)

#include <CL/opencl.hpp>

#include <common/chrono.hpp>
#include <common/error/opencl_error.hpp>
#include <common/kernel_generator/opencl.hpp>
#include <common/log/log.hpp>


void common::KernelGeneratorOpenCL::clear()
{
    clKernel = nullptr;
    clProgram = nullptr;

    common::KernelGenerator::clear();
}


bool common::KernelGeneratorOpenCL::build(cl::Device* const clDevice, cl::Context* const clContext)
{
    try
    {
        ////////////////////////////////////////////////////////////////////////
        std::string    fullSource;
        common::Chrono chrono;

        ////////////////////////////////////////////////////////////////////////
        chrono.start();

        ////////////////////////////////////////////////////////////////////////
        if (false == defines.empty())
        {
            fullSource += "///////////////////////////////////////////////////////////////////////////////";
            fullSource += "\n";
            for (auto const& kv : defines)
            {
                fullSource += "#define " + kv.first + " " + kv.second + "\n";
            }
            fullSource += "///////////////////////////////////////////////////////////////////////////////";
            fullSource += "\n";
            fullSource += "\n";
        }

        ////////////////////////////////////////////////////////////////////////
        if (false == includes.empty())
        {
            fullSource += "///////////////////////////////////////////////////////////////////////////////";
            fullSource += "\n";
            for (auto const& pathInclude : includes)
            {
                fullSource += "#include \"";
                fullSource += pathInclude;
                fullSource += "\"";
                fullSource += "\n";
            }
            fullSource += "///////////////////////////////////////////////////////////////////////////////";
            fullSource += "\n";
            fullSource += "\n";
        }

        ////////////////////////////////////////////////////////////////////////
        fullSource += sourceCode;

        ////////////////////////////////////////////////////////////////////////
        writeKernelInFile(fullSource, ".cl");

        ////////////////////////////////////////////////////////////////////////
        clProgram = cl::Program(*clContext, fullSource);

        ////////////////////////////////////////////////////////////////////////
        // -cl-fast-relaxed-math implies -cl-no-signed-zeros and -cl-mad-enable
        // per the OpenCL spec, so those are not added separately.
        compileFlags += " -cl-fast-relaxed-math";

        // Kernel-language version is a device capability, not a platform: query
        // it at runtime so Apple (OpenCL C 1.2) and modern AMD (3.0) each get the
        // highest -cl-std they support. Falls back to 1.2 if the version string
        // cannot be parsed.
        std::string const deviceClcVersion{ clDevice->getInfo<CL_DEVICE_OPENCL_C_VERSION>() };
        char const* clStd{ " -cl-std=CL1.2" };
        if (std::string::npos != deviceClcVersion.find("OpenCL C 3."))
        {
            clStd = " -cl-std=CL3.0";
        }
        else if (std::string::npos != deviceClcVersion.find("OpenCL C 2."))
        {
            clStd = " -cl-std=CL2.0";
        }
        compileFlags += clStd;

        // -O3 is a clang/AMD extension, not a standard OpenCL build option; Apple's
        // 1.2 compiler rejects it with CL_INVALID_BUILD_OPTIONS. Use the standard
        // -cl-opt-disable for debug everywhere, and keep -O3 only off-Apple.
        std::string const platformVendor{
            cl::Platform(clDevice->getInfo<CL_DEVICE_PLATFORM>()).getInfo<CL_PLATFORM_VENDOR>() };
        bool const isApplePlatform{ std::string::npos != platformVendor.find("Apple") };
#if defined(__DEBUG)
        compileFlags += " -cl-opt-disable";
#else
        if (false == isApplePlatform)
        {
            compileFlags += " -O3";
        }
#endif
        compileFlags += " -I ./";

        ////////////////////////////////////////////////////////////////////////
        clProgram.build(*clDevice, compileFlags.c_str());
        clKernel = cl::Kernel(clProgram, kernelName.c_str());

        ////////////////////////////////////////////////////////////////////////
        chrono.stop();
        logInfo() << "Build kernel " << kernelName << " in " << chrono.elapsed(common::CHRONO_UNIT::MS) << "ms";

        ////////////////////////////////////////////////////////////////////////
        sourceCode.clear();
    }
    catch (cl::BuildError const& clErr)
    {
        auto const clBuildStatus{ clProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*clDevice) };
        logErr() << "(" << __FUNCTION__ << ")"
                 << "{" << openclShowError(clErr.err()) << "}"
                 << " -> " << kernelName << "\n"
                 << clBuildStatus;
        return false;
    }
    catch (cl::Error const& clErr)
    {
        logErr() << "Fail to build kernel " << kernelName;
        OPENCL_EXCEPTION_ERROR_SHOW(__FUNCTION__, clErr);
        return false;
    }

    ////////////////////////////////////////////////////////////////////////
    built = true;
    return built;
}

#endif
