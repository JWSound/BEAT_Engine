#!/bin/bash
# Egyszeri telepítés: a futtató a háttérben indul bejelentkezéskor, és amíg fut, nem engedi elaludni a gépet.
KIT="$HOME/BoundaryLab/_beat_kit"; P="$HOME/Library/LaunchAgents/local.beat.hmat-runner.plist"
chmod +x "$KIT/runner/watch.sh"; mkdir -p "$HOME/Library/LaunchAgents"
cat > "$P" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>local.beat.hmat-runner</string>
<key>ProgramArguments</key><array><string>/usr/bin/caffeinate</string><string>-i</string><string>/bin/bash</string><string>$KIT/runner/watch.sh</string></array>
<key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
<key>StandardErrorPath</key><string>$KIT/results/runner.err</string>
</dict></plist>
PL
launchctl unload "$P" 2>/dev/null; launchctl load "$P" && echo "Futtató telepítve és elindítva. Leállítás: launchctl unload $P"
