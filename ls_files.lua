function lsFiles(dir)
  local files = {}
  for f in io.popen("ls -F " .. dir .. " | grep -v '/'"):lines() do
    table.insert(files, f)
  end
  return files
end
