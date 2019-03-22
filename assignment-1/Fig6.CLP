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
    (assert(Z)))

(defrule r2
    (X)
    (B)
    (E)
    =>
    (assert(Y)))

(defrule r3
    (A)
    =>
    (assert(X)))

(defrule r4
    (C)
    =>
    (assert (L)))

(defrule r5
    (L)
    (M)
    =>
    (assert(N)))