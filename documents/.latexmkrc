# Enable shell-escape for minted package (works with all TeX engines)
set_tex_cmds('-shell-escape %O %S');

# Use xelatex as the PDF generator
$pdf_mode = 5;  # 5 = xelatex

# Explicit commands with shell-escape (fallback if set_tex_cmds doesn't work)
$xelatex = 'xelatex -shell-escape -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$pdflatex = 'pdflatex -shell-escape -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$lualatex = 'lualatex -shell-escape -synctex=1 -interaction=nonstopmode -file-line-error %O %S';

# Additional files to clean with latexmk -c
$clean_ext = "synctex.gz synctex.gz(busy) run.xml";

# Clean minted cache directory
push @generated_exts, '_minted-%R';
