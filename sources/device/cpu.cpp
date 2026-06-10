#if defined(CPU_ENABLE)

#include <common/log/log.hpp>
#include <device/cpu.hpp>


bool device::DeviceCpu::initialize()
{
    logInfo() << "Initialize device CPU";
    return true;
}


void device::DeviceCpu::cleanUp()
{
    logInfo() << "Clean up device CPU";
}

#endif
