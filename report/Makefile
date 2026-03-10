REPORT_DIR = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
TEX = report.tex
PDF = report.pdf

.PHONY: pdf check clean

pdf:
	docker run --rm \
		-v $(REPORT_DIR):/workspace \
		-w /workspace \
		texlive/texlive:latest \
		latexmk -pdf -interaction=nonstopmode $(TEX)

check:
	@echo "Checking report.tex structure..."
	@grep -c "\\\\section" $(REPORT_DIR)$(TEX) && echo "Sections: OK"
	@grep -c "\\\\placeholder" $(REPORT_DIR)$(TEX) && echo "Placeholders defined"
	@test -f $(REPORT_DIR)references.bib && echo "references.bib: OK"

clean:
	docker run --rm \
		-v $(REPORT_DIR):/workspace \
		-w /workspace \
		texlive/texlive:latest \
		latexmk -C $(TEX)
