package main

import (
	"fmt"
	"os"
	"sync"
	"time"
)

const (
	LogMaxBytes = 512 * 1024 // 512 KB
	LogBackups  = 3
)

type RotatingLogger struct {
	mu   sync.Mutex
	path string
}

var globalLogger = &RotatingLogger{
	path: "superrelay.log",
}

func InitLogger(path string) {
	if path != "" {
		globalLogger.path = path
	}
}

func (l *RotatingLogger) rotate() {
	fi, err := os.Stat(l.path)
	if err != nil || fi.Size() < LogMaxBytes {
		return
	}

	oldest := fmt.Sprintf("%s.%d", l.path, LogBackups)
	_ = os.Remove(oldest)

	for i := LogBackups - 1; i >= 1; i-- {
		src := fmt.Sprintf("%s.%d", l.path, i)
		dst := fmt.Sprintf("%s.%d", l.path, i+1)
		_ = os.Rename(src, dst)
	}

	_ = os.Rename(l.path, fmt.Sprintf("%s.1", l.path))
}

func (l *RotatingLogger) Logf(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	ts := time.Now().Format("15:04:05")
	line := fmt.Sprintf("[%s] %s\n", ts, msg)

	l.mu.Lock()
	defer l.mu.Unlock()

	l.rotate()

	// Write to file
	f, err := os.OpenFile(l.path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err == nil {
		_, _ = f.WriteString(line)
		_ = f.Close()
	}

	// Also print to console
	fmt.Print(line)
}

func LogGateway(format string, args ...interface{}) {
	globalLogger.Logf(format, args...)
}
