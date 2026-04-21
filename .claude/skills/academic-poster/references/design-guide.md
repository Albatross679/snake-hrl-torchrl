# Academic Poster Design Guide

## Table of Contents
- [Layout Rules](#layout-rules)
- [Typography](#typography)
- [Color](#color)
- [Content Guidelines](#content-guidelines)
- [Figures](#figures)
- [tikzposter Themes Reference](#tikzposter-themes-reference)
- [Custom Color Palette](#custom-color-palette)
- [Common Pitfalls](#common-pitfalls)

## Layout Rules

- **20/40/40 rule:** ~20% text, ~40% figures/visuals, ~40% whitespace/margins
- **800 words max.** If it can't fit, it shouldn't be on the poster
- **3-column landscape A0** is the default for CS/ML posters
- Column widths: equal thirds (0.33/0.34/0.33) or weighted center (0.25/0.50/0.25)
- Column gutters: minimum 2-3cm
- One idea per block. Don't combine unrelated content

## Typography

| Level | Element | Font Size (A0) | Weight |
|-------|---------|----------------|--------|
| 1 | Title | 80-100pt | Bold |
| 2 | Section headings | 50-60pt | Bold |
| 3 | Body text | 30-36pt | Regular |
| 4 | Captions/refs | 24-28pt | Regular/Italic |

- Max 2 font families. Sans-serif for headings, serif or sans for body
- Minimum 24pt for any text on A0
- Line spacing 1.2-1.5x for body
- Left-align body text, center only titles
- No ALL CAPS for body text

## Color

- 3-5 colors maximum
- One primary (headings, accents), one secondary (highlights, borders), one background
- Same color = same semantic meaning throughout
- Avoid red/green only distinctions (colorblind-safe)
- Avoid light yellows, light cyans, neon colors (print poorly)

## Content Guidelines

### Include
| Section | Column | Content |
|---------|--------|---------|
| Title | Full width | Title, author, institution |
| Problem & Motivation | Left | What and why, 2-3 sentences + example |
| Dataset & Evaluation | Left | Data description, metrics |
| Methods (1 block each) | Center | Architecture/approach, key hyperparams |
| Results | Right | Simplified table + figures |
| Key Findings | Right | 3-5 takeaway bullets |
| References | Right bottom | 3-5 citations, small font |

### Exclude (differs from paper)
- Abstract (the poster IS the abstract)
- Related work section
- Detailed hyperparameter tables
- Full mathematical derivations
- Exhaustive results tables
- Long prose paragraphs

### Writing Style
- Bullet points over paragraphs, 3-6 per block
- Active voice, present tense
- Quantitative claims ("85.3% F1" not "good performance")
- Include example input/output pairs where possible

## Figures

- Use vector (PDF/SVG) for all plots — perfect at any print size
- Minimum 15cm wide on poster for readability from 1.5m
- Label directly with arrows/callouts, not lengthy captions
- Enlarge axis labels beyond paper-figure sizes
- Match figure color palette to poster palette
- Use `\includegraphics[width=0.9\linewidth]{../media/figure.pdf}`

## tikzposter Themes Reference

### Themes
| Theme | Style |
|-------|-------|
| Default | Clean, minimal blocks |
| Wave | Wavy title bar, modern |
| Basic | Simple bordered blocks |
| Simple | Minimal decoration |
| Board | Bulletin-board aesthetic |
| Envelope | Folded-corner blocks |

### Built-in Color Styles
| Style | Palette |
|-------|---------|
| Default | Blue/orange |
| Denmark | Red/white (high contrast) |
| Britain | Blue/red/white |
| Sweden | Blue/yellow |
| Australia | Green/gold |

## Custom Color Palette

```latex
\definecolorstyle{MyStyle}{
  \definecolor{colorOne}{HTML}{XXXXXX}   % primary
  \definecolor{colorTwo}{HTML}{XXXXXX}   % secondary
  \definecolor{colorThree}{HTML}{XXXXXX} % background
}{
  \colorlet{backgroundcolor}{colorThree}
  \colorlet{framecolor}{colorTwo}
  \colorlet{titlebgcolor}{colorOne}
  \colorlet{titlefgcolor}{white}
  \colorlet{blocktitlebgcolor}{colorOne}
  \colorlet{blocktitlefgcolor}{white}
  \colorlet{blockbodybgcolor}{white}
  \colorlet{blockbodyfgcolor}{black}
}
\usecolorstyle{MyStyle}
```

## Common Pitfalls

1. **Text overload** — >1000 words, prose paragraphs. Convert to 3-5 bullet points per block
2. **Tiny fonts** — Body <24pt. Always verify readability at 1.5m viewing distance
3. **Low-res figures** — Screen PNGs at 72 DPI. Use vector PDFs instead
4. **Color print mismatch** — RGB neons/pastels shift in CMYK print. Use high-contrast
5. **Inconsistent style** — Figures with different fonts/colors. Use one matplotlib style for all
6. **Missing "so what?"** — Results without takeaway. Always include Key Findings block
7. **Shrunk paper** — Copying paper verbatim. Poster is NOT a mini-paper
