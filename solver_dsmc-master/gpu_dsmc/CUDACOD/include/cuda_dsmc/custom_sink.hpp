#ifndef CUSTOM_SINK_HPP_INCLUDED
#define CUSTOM_SINK_HPP_INCLUDED 1

struct CustomSink {
// Linux xterm color
  enum FG_Color {YELLOW = 33, RED = 31, GREEN = 32, WHITE = 97, CYAN = 36};

  FG_Color GetColor(const LEVELS level) const {
     if (level.value == WARNING.value) { return YELLOW; }
     if (level.value == DEBUG.value) { return GREEN; }
     if (g3::internal::wasFatal(level)) { return RED; }

     return CYAN;
  }

  void ReceiveLogMessage(g3::LogMessageMover logEntry) {
     auto level = logEntry.get()._level;
     auto color = GetColor(level);

     std::cout << "\033[" << color << "m"
               << logEntry.get().toString() << "\033[m" << std::endl;
  }
};

// CUSTOM_SINK_HPP_INCLUDED
#endif
