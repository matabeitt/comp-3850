(clear)
(defrule ask-question
    => (printout t "Starter turning (yes or no)" crlf)
       (bind ?ans (read))
       (assert (turning ?ans)))

(defrule ask-petrol 
    (turning yes) 
    => 
    (printout t "Got any petrol? (yes or no)" crlf)
    (bind ?ans (read))
    (assert (petrol ?ans)))

(defrule ask-lights
    (turning no)
    =>
    (printout t "Are your lights working? (yes or no)" crlf)
    (bind ?ans (read))
    (assert (lights ?ans)))

(defrule ask-solenoid
    (turning no)
    (lights yes)
    =>
    (printout t "Is there a solenoid click? (yes or no)" crlf)
    (bind ?ans (read))
    (assert (solenoid ?ans)))

(defrule ask-terminals
    (turning no)
    (lights yes)
    (solenoid yes)
    =>
    (printout t "Are your terminals clean? (yes or no)" crlf)
    (bind ?ans (read))
    (assert (terminals-clean ?ans)))

(defrule ask-solenoid-fuse
    (turning no)
    (lights yes)
    (solenoid no)
    =>
    (printout t "Is your solenoid fuse ok? (yes or no)" crlf)
    (bind ?ans (read))
    (assert (solenoid-fuse ?ans)))

(reset)
(run)