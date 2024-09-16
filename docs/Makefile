SPHINXBUILD = sphinx-build
BUILDDIR = build
SPHINXOPTS =
ALLSPHINXOPTS = $(SPHINXOPTS) .

.PHONY: clean html linkcheck examples docs

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf api
	rm -rf examples

html:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

examples:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

docs: html linkcheck

