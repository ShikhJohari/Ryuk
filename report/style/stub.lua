-- A `::: stub` div marks a passage not written yet; its text names the ticket that writes it.
-- In Typst it renders through the template's `#stub` function (style/typst-template.typ), so
-- stubs are set apart from finished prose. Other formats keep the div as it is.
--
-- A target whose metadata sets `stubs: forbidden` must have none (#32, the audit's T17): its
-- render fails, naming every stub left. report.qmd sets it, so the full report, which includes
-- every section, cannot be built with a passage unwritten. progress.qmd does not: a partial
-- render may show work in progress, and any stub it shows is in a section the full report
-- includes too, so `quarto render report` still fails on it.

local forbidden = false
local left = pandoc.List()

local function read_meta(meta)
  forbidden = meta.stubs ~= nil and pandoc.utils.stringify(meta.stubs) == "forbidden"
end

-- The stub's first 80 characters, to say which one it is.
local function opening_words(el)
  local text = table.concat(el.content:map(pandoc.utils.stringify), " ")
  if utf8.len(text) > 80 then
    text = text:sub(1, utf8.offset(text, 78) - 1) .. "..."
  end
  return text
end

local function render_stub(el)
  if not el.classes:includes("stub") then
    return nil
  end
  if forbidden then
    left:insert(opening_words(el))
    return nil
  end
  if not quarto.doc.is_format("typst") then
    return nil
  end
  local blocks = pandoc.List({ pandoc.RawBlock("typst", "#stub[") })
  blocks:extend(el.content)
  blocks:insert(pandoc.RawBlock("typst", "]"))
  return blocks
end

-- Quarto logs an error raised in a filter and carries on, so the render would still succeed;
-- exiting is what fails it, after every stub has been named.
local function fail_on_stubs(doc)
  if #left == 0 then
    return nil
  end
  local source = quarto.doc.input_file or "this document"
  io.stderr:write(
    "ERROR: " .. #left .. " `::: stub` block" .. (#left == 1 and " is" or "s are") .. " left in "
      .. source .. ", which sets `stubs: forbidden`. Write each passage, or remove its stub:\n"
  )
  for _, words in ipairs(left) do
    io.stderr:write('  - "' .. words .. '"\n')
  end
  io.stderr:flush()
  os.exit(1)
end

-- Three passes: the metadata is read before any div is visited, and every div before failing.
return {
  { Meta = read_meta },
  { Div = render_stub },
  { Pandoc = fail_on_stubs },
}
