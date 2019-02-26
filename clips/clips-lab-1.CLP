(deffacts  database
	(male tom)
	(male tim)
	(female jane)
	(child tom tim)
	(child tim jane)
)

(defrule is-father
	(child ?x ?y) (male ?x) =>
	(assert (father ?y ?x))
	(printout t ?y " is " ?x " father." crlf)
)

(defrule is-married
	(child ?x ?y)
	(child ?x ?z) =>
	if (not (eq ?y ?z)
		then (assert (married ?y ?z))
		(printout t ?y ?z " are married." crlf)
	)
)

;;; Implementation of a Finite State Machine that accepts an even number of 'A'.

;;; Implementation of a (template ...)