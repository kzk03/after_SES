$latex = 'uplatex -synctex=1 -halt-on-error -file-line-error %O %S';
$bibtex = 'upbibtex';
$biber = 'biber --bblencoding=utf8 -u -U --output_safechars';
$dvipdf = 'dvipdfmx -o %D %S';
$makeindex = 'mendex -U %O -o %D %S';
$pdf_mode = 3;
