# Restores the per-user registry keys changed by Register-NapalmFileIcons.ps1.
[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$backupPath = Join-Path (Join-Path $env:LOCALAPPDATA 'Napalm') 'file-icons-backup.json'
if (-not (Test-Path -LiteralPath $backupPath)) {
    throw "Icon backup not found: $backupPath"
}
$backup = Get-Content -LiteralPath $backupPath -Raw | ConvertFrom-Json

if ($PSCmdlet.ShouldProcess('HKCU\Software\Classes', 'Restore Napalm file-type icons')) {
    foreach ($property in $backup.PSObject.Properties) {
        $extension = $property.Name
        if ($extension -notin @('.napalm', '.fire', '.burning')) {
            throw "Unexpected extension in backup: $extension"
        }
        $regPath = "Registry::HKEY_CURRENT_USER\Software\Classes\$extension\DefaultIcon"
        if ($property.Value.Exists) {
            if (-not (Test-Path -LiteralPath $regPath)) {
                New-Item -Path $regPath -Force | Out-Null
            }
            if ($null -eq $property.Value.Value) {
                $key = Get-Item -LiteralPath $regPath
                $key.DeleteValue('', $false)
            } else {
                Set-Item -LiteralPath $regPath -Value $property.Value.Value
            }
        } elseif (Test-Path -LiteralPath $regPath) {
            Remove-Item -LiteralPath $regPath -Recurse -Force
        }
    }
    Remove-Item -LiteralPath $backupPath -Force
    Add-Type -Namespace Napalm -Name Shell -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("shell32.dll")]
public static extern void SHChangeNotify(uint eventId, uint flags, System.IntPtr item1, System.IntPtr item2);
'@
    [Napalm.Shell]::SHChangeNotify(0x08000000, 0, [IntPtr]::Zero, [IntPtr]::Zero)
    Write-Output 'Previous file-type icon mappings restored.'
}
