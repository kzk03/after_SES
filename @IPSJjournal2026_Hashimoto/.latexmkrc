$latex = 'platex -kanji=utf8 -synctex=1 -interaction=nonstopmode -file-line-error %O %S';              
$bibtex = 'pbibtex -kanji=utf8';                      
$dvipdf = 'dvipdfmx %O -o %D %S';                 
$makeindex = 'mendex %O -o %D %S';            
$pdf_mode = 3;