const openCitation = (citation) => {
  const url = `/files/${citation.source}#page=${citation.page}`;
  setPdfUrl(url);
  setHighlightText(citation.highlight_text);
};
