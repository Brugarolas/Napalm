# Windows Explorer icons

File icons in Explorer are independent of VS Code file icon themes. These
scripts register **per-user** `DefaultIcon` keys for `.napalm`, `.fire`, and
`.burning`, without replacing any selected default editor or application.

Prepare your own three `.ico` files, then run (from this directory):

```powershell
.\Register-NapalmFileIcons.ps1 -NapalmIcon 'C:\art\napalm.ico' `
    -FireIcon 'C:\art\fire.ico' -BurningIcon 'C:\art\burning.ico'
```

The script creates a backup at `%LOCALAPPDATA%\Napalm\file-icons-backup.json`.
To restore the previous settings:

```powershell
.\Restore-NapalmFileIcons.ps1
```

Windows may prefer the icon supplied by the current default application for
some extensions. In that case a proper application ProgID / installer
registration may be required. These scripts do not forcibly change the
user's default application and make no claim that an arbitrary icon will
override every shell association. Explorer can cache icons until refreshed.
