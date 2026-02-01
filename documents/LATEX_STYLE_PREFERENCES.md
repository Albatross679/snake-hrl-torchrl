# LaTeX Style Preferences

This document outlines the LaTeX formatting preferences for reports and academic documents.

## Page Layout

- **Top margin**: 0.6 inches
- **Bottom margin**: 0.6 inches
- **Horizontal margins**: 0.75 inches
- **Include header and footer in layout**: Yes
- **Header height**: 14pt
- **Footer skip**: 0.25 inches

```latex
\usepackage[top=0.6in, bottom=0.6in, hmargin=0.75in, includehead, includefoot, headheight=14pt, footskip=0.25in]{geometry}
```

## Fonts

- **Main font**: Palatino
- **URLs/Links**: Sans-serif (Helvetica) using `\textsf{}`

```latex
\usepackage{palatino}
\usepackage[T1]{fontenc}
```

## Header and Footer

- **Header**: Class name and number centered on every page (including first page)
- **Footer**: GitHub link (without `https://`) centered, page number on the right
- **Header rule**: 0.4pt line below header
- **Footer rule**: None

```latex
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[C]{\small Class Name and Number}
\fancyfoot[C]{\small \textsf{\href{https://github.com/...}{github.com/...}}}
\fancyfoot[R]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}

% Apply to first page as well
\fancypagestyle{plain}{
  \fancyhf{}
  \fancyhead[C]{\small Class Name and Number}
  \fancyfoot[C]{\small \textsf{\href{https://github.com/...}{github.com/...}}}
  \fancyfoot[R]{\thepage}
  \renewcommand{\headrulewidth}{0.4pt}
}
```

## Title Formatting

- **Title position**: Reduced space above title (`\droptitle{-5em}`)
- **Title size**: `\LARGE\bfseries`
- **Subtitle**: Include assignment number below main title
- **Author format**: Name and email on same line with email in parentheses
- **Spacing**: Tight spacing between title elements

```latex
\usepackage{titling}
\setlength{\droptitle}{-5em}
\pretitle{\begin{center}\LARGE\bfseries}
\posttitle{\end{center}\vspace{-1em}}
\preauthor{\begin{center}}
\postauthor{\end{center}\vspace{-2em}}
\predate{}
\postdate{}

\title{Main Title\\[0.3em]\large Assignment Number}
\author{Name (\href{mailto:email@example.com}{email@example.com})}
\date{}
```

## Paragraph Formatting

- **Indentation**: None
- **Paragraph spacing**: 0.5em between paragraphs
- **Line spacing**: 1.1 stretch

```latex
\usepackage{setspace}
\setstretch{1.1}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5em}
```

## Section Spacing

- **Sections**: 1em before, 0.5em after
- **Subsections**: 0.8em before, 0.3em after

```latex
\usepackage{titlesec}
\titlespacing*{\section}{0pt}{1em}{0.5em}
\titlespacing*{\subsection}{0pt}{0.8em}{0.3em}
```

## List Formatting

- **Bullet points**: Compact spacing with minimal gaps

```latex
\usepackage{enumitem}
\setlist[itemize]{nosep, topsep=0.3em, partopsep=0pt, parsep=0pt, itemsep=0.2em}
```

## Hyperlinks

- **Link styling**: Hidden (no colored boxes or underlines)
- **Links remain clickable**

```latex
\usepackage[hidelinks]{hyperref}
```

## Tables

- **Position**: Force exact placement with `[H]` specifier
- **Style**: Use `booktabs` for professional rules

```latex
\usepackage{float}
\usepackage{booktabs}

\begin{table}[H]
\centering
\begin{tabular}{@{}lcc@{}}
\toprule
...
\bottomrule
\end{tabular}
\caption{...}
\end{table}
```

## Build Configuration

- **Engine**: XeLaTeX
- **Build tool**: latexmk (handles multiple passes automatically)

`.latexmkrc` configuration (place in the LaTeX document folder, e.g. `documents/.latexmkrc`):
```perl
$pdflatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$pdf_mode = 1;
$clean_ext = "synctex.gz synctex.gz(busy) run.xml xdv";
```

## Cleanup and Explorer Hygiene

- **Auto-clean**: Enable `latex-workshop.latex.autoClean.run` = `onBuilt`
- **Cleaned files**: Include `.xdv` (XeLaTeX intermediate) in clean list
- **Explorer**: Hide common LaTeX aux files to reduce clutter

## Complete Preamble Template

```latex
\documentclass[11pt]{article}

% Font
\usepackage{palatino}
\usepackage[T1]{fontenc}

% Page layout
\usepackage[top=0.6in, bottom=0.6in, hmargin=0.75in, includehead, includefoot, headheight=14pt, footskip=0.25in]{geometry}
\usepackage{setspace}
\setstretch{1.1}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5em}

% Header and footer
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[C]{\small Class Name}
\fancyfoot[C]{\small \textsf{\href{https://github.com/user/repo}{github.com/user/repo}}}
\fancyfoot[R]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}
\fancypagestyle{plain}{\fancyhf{}\fancyhead[C]{\small Class Name}\fancyfoot[C]{\small \textsf{\href{https://github.com/user/repo}{github.com/user/repo}}}\fancyfoot[R]{\thepage}\renewcommand{\headrulewidth}{0.4pt}}

% Standard packages
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}

% Compact spacing
\usepackage{titlesec}
\titlespacing*{\section}{0pt}{1em}{0.5em}
\titlespacing*{\subsection}{0pt}{0.8em}{0.3em}

\usepackage{enumitem}
\setlist[itemize]{nosep, topsep=0.3em, partopsep=0pt, parsep=0pt, itemsep=0.2em}

% Title formatting
\usepackage{titling}
\setlength{\droptitle}{-5em}
\pretitle{\begin{center}\LARGE\bfseries}
\posttitle{\end{center}\vspace{-1em}}
\preauthor{\begin{center}}
\postauthor{\end{center}\vspace{-2em}}
\predate{}
\postdate{}
```
