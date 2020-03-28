FILE(REMOVE_RECURSE
  "CMakeFiles/clean-cmake"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/clean-cmake.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
