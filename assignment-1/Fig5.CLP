; Ensure CLIPS Console is (clear)
; Load the File
; Ensure Knowledge Base is empty (reset)
; (run) the File

(deffacts database
    (A)
    (B)
    (C)
    (D)
    (E))

(defrule r1
    (Y)
    (D)
    =>
    (assert (Z)))

(defrule r2
    (X)
    (B)
    (E)
    =>
    (assert (Y)))

(defrule r3
    (A)
    =>
    (assert (X)))