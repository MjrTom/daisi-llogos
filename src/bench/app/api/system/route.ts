// System info endpoint — CPU, GPU, RAM for benchmark context

import { execSync } from "child_process";

export async function GET() {
  const info: Record<string, string> = {};

  try {
    // CPU
    const cpu = execSync(
      'powershell -Command "(Get-CimInstance Win32_Processor).Name"',
      { encoding: 'utf8' }
    ).trim();
    info.cpu = cpu || 'Unknown';
  } catch { info.cpu = 'Unknown'; }

  try {
    // RAM
    const ramBytes = execSync(
      'powershell -Command "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB, 1)"',
      { encoding: 'utf8' }
    ).trim();
    info.ram = `${ramBytes} GB`;
  } catch { info.ram = 'Unknown'; }

  try {
    // GPU
    const gpu = execSync('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader', { encoding: 'utf8' }).trim();
    const [name, vram, driver] = gpu.split(', ').map(s => s.trim());
    info.gpu = name || 'Unknown';
    info.vram = vram || 'Unknown';
    info.gpuDriver = driver || 'Unknown';
  } catch { info.gpu = 'No NVIDIA GPU'; }

  try {
    // OS
    const os = execSync(
      'powershell -Command "(Get-CimInstance Win32_OperatingSystem).Caption"',
      { encoding: 'utf8' }
    ).trim();
    info.os = os || 'Unknown';
  } catch { info.os = 'Unknown'; }

  try {
    // .NET
    const dotnet = execSync('dotnet --version', { encoding: 'utf8' }).trim();
    info.dotnet = dotnet;
  } catch { info.dotnet = 'Unknown'; }

  return Response.json(info);
}
