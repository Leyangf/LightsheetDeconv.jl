using LightsheetDeconv
using Documenter

DocMeta.setdocmeta!(LightsheetDeconv, :DocTestSetup, :(using LightsheetDeconv); recursive=true)

makedocs(;
    modules=[LightsheetDeconv],
    authors="Yueyang Feng<yueyang.f.de@gmail.com>",
    sitename="LightsheetDeconv.jl",
    format=Documenter.HTML(;
        canonical="https://Leyangf.github.io/LightsheetDeconv.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Leyangf/LightsheetDeconv.jl",
    devbranch="master",
)
