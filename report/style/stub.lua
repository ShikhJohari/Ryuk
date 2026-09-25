-- A `::: stub` div marks a passage not written yet; its text names the ticket that writes it.
-- In Typst it renders through the template's `#stub` function (style/typst-template.typ), so
-- stubs are set apart from finished prose. Other formats keep the div as it is.

function Div(el)
  if not el.classes:includes("stub") or not quarto.doc.is_format("typst") then
    return nil
  end
  local blocks = pandoc.List({ pandoc.RawBlock("typst", "#stub[") })
  blocks:extend(el.content)
  blocks:insert(pandoc.RawBlock("typst", "]"))
  return blocks
end
