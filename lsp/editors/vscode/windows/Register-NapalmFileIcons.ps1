# Registers only per-user file-type icons, without changing default applications.
[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param(
    [Parameter(Mandatory = $true)][string] $NapalmIcon,
    [Parameter(Mandatory = $true)][string] $FireIcon,
    [Parameter(Mandatory = $true)][string] $BurningIcon
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$requestedIcons = [ordered]@{
    '.napalm' = $NapalmIcon
    '.fire' = $FireIcon
    '.burning' = $BurningIcon
}

$installed = @{}
foreach ($entry in $requestedIcons.GetEnumerator()) {
    $resolved = (Resolve-Path -LiteralPath $entry.Value -ErrorAction Stop).ProviderPath
    if ([System.IO.Path]::GetExtension($resolved).ToLowerInvariant() -ne '.ico') {
        throw "Windows Explorer icon must be a .ico file: $resolved"
    }
    $installed[$entry.Key] = $resolved
}

$backupDir = Join-Path $env:LOCALAPPDATA 'Napalm'
$backupPath = Join-Path $backupDir 'file-icons-backup.json'
if (Test-Path -LiteralPath $backupPath) {
    throw "A backup already exists at $backupPath. Restore it before running this script again."
}

$previous = [ordered]@{}
foreach ($extension in $installed.Keys) {
    $regPath = "Registry::HKEY_CURRENT_USER\Software\Classes\$extension\DefaultIcon"
    $key = Get-Item -LiteralPath $regPath -ErrorAction SilentlyContinue
    if ($null -eq $key) {
        $previous[$extension] = @{ Exists = $false; Value = $null }
    } else {
        $previous[$extension] = @{ Exists = $true; Value = $key.GetValue('', $null) }
    }
}

if ($PSCmdlet.ShouldProcess('HKCU\Software\Classes', 'Register Napalm file-type icons')) {
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    $previous | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $backupPath -Encoding utf8

    foreach ($extension in $installed.Keys) {
        $regPath = "Registry::HKEY_CURRENT_USER\Software\Classes\$extension\DefaultIcon"
        if (-not (Test-Path -LiteralPath $regPath)) {
            New-Item -Path $regPath -Force | Out-Null
        }
        Set-Item -LiteralPath $regPath -Value "`"$($installed[$extension])`",0"
    }

    Add-Type -Namespace Napalm -Name Shell -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("shell32.dll")]
public static extern void SHChangeNotify(uint eventId, uint flags, System.IntPtr item1, System.IntPtr item2);
'@
    [Napalm.Shell]::SHChangeNotify(0x08000000, 0, [IntPtr]::Zero, [IntPtr]::Zero)
    Write-Output "Registered icon associations. Backup: $backupPath"
}
