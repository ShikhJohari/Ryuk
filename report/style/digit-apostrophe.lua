-- Typst reads an apostrophe straight after a digit as a prime, so "#9's" would print "#9′s".
-- Pandoc hands apostrophes to Typst as ASCII quotes for it to make smart, so these ones are
-- passed through as a literal right single quote instead, which Typst prints as it is.

local APOSTROPHE = "’"

function Str(el)
  if not quarto.doc.is_format("typst") then
    return nil
  end
  local text = el.text:gsub("(%d)'", "%1" .. APOSTROPHE)
  if not text:find("%d" .. APOSTROPHE) then
    return nil
  end
  local out, start = pandoc.List(), 1
  while true do
    local s, e = text:find("%d" .. APOSTROPHE, start)
    if not s then
      break
    end
    out:insert(pandoc.Str(text:sub(start, s)))
    out:insert(pandoc.RawInline("typst", APOSTROPHE))
    start = e + 1
  end
  if start <= #text then
    out:insert(pandoc.Str(text:sub(start)))
  end
  return out
end
