$latex    = 'platex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$bibtex   = 'pbibtex';
$dvipdf   = 'dvipdfmx %O -o %D %S';
$makeindex = 'mendex %O -o %D %S';
$pdf_mode = 3;  # dvi -> pdf
