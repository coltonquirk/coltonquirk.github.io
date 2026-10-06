using CairoMakie

inch = 96
pt = 4/3
cm = inch / 2.54

dark_latexfonts = merge(theme_dark(), theme_latexfonts())

function logo(;length=1inch, fontsize=24pt)
    fig = Figure(size=(length, length), fontsize=fontsize)
    ax = Axis(fig[1,1])

    hidedecorations!()
    hidespines!()

    text!(
        0, 0,
        text="CQ",
        align=(:center, :center),
        color=:white
    )

    fig
end

fontsize=34pt
logo(fontsize=fontsize)

with_theme(dark_latexfonts) do
    fig = logo(fontsize=fontsize)
    save("logo.png", fig, px_per_unit = 100/inch)
end
