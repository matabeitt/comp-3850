(defrule stimulus-situation-verbal
    (or
        (environment papers)
        (environment manuals)
        (environment documents)
        (environment textbooks)
    )
    =>
    (assert(stimulus-situation verbal))
)

(defrule stimulus-situation-visual
    (or
        (environment pictures)
        (environment illustrations)
        (environment photographs)
        (environment diagrams)
    )
    =>
    (assert(stimulus-situation visual))
)

(defrule stimulus-situation-physical
    (or
        (environment machines)
        (environment buildings)
        (environment tools)
    )
    =>
    (assert(stimulus-situation physical))
)

(defrule stimulus-situation-symbolic
    (or
        (environment numbers)
        (environment formulas)
        (environment programs)
    )
    =>
    (assert(stimulus-situation symbolic))
)

(defrule stimulus-response-oral
    (or
        (job lecturing)
        (job advising)
        (job counselling)
    )
    =>
    (assert(stimulus-response oral))
)

(defrule stimulus-response-hands-on
    (or
        (job building)
        (job repairing)
        (job troubleshooting)
    )
    =>
    (assert(stimulus-response hands-on))
)

(defrule stimulus-response-documented
    (or
        (job writing)
        (job typing)
        (job drawing)
    )
    =>
    (assert(stimulus-response documented))
)

(defrule stimulus-response-analytical
    (or
        (job evaluating)
        (job reasoning)
        (job investigating)
    )
    =>
    (assert(stimulus-response analytical))
)

(defrule medium-workshop
    (and
        (stimulus-situation physical)
        (stimulus-response hands-on)
        (feedback yes)
    )
    =>
    (assert(medium workshop))
)

(defrule medium-lecture
    (or
        (and
            (stimulus-situation symbolic)
            (stimulus-response analytical)
            (feedback yes)
        )
        (and
            (stimulus-situation visual)
            (stimulus-response oral)
            (feedback yes)
        )
        (and
            (stimulus-situation verbal)
            (stimulus-response analytical)
            (feedback yes)
        )
    )
    =>
    (assert(medium lecture))
)

(defrule medium-videocassette
    (and
        (stimulus-situation visual)
        (stimulus-response documented)
        (feedback yes)
    )
    =>
    (assert(medium videocassette))
)

(defrule medium-roleplay
    (and
        (stimulus-situation verbal)
        (stimulus-response oral)
        (feedback yes)
    )
    =>
    (assert(medium roleplay))
)

(defrule ask-environment
    =>
    (printout t "What is the environment?" crlf)
    (bind ?ans (read))
    (assert (environment (sym-cat ?ans)))
)

(defrule ask-job
    =>
    (printout t "What is your job?" crlf)
    (bind ?ans (read))
    (assert (job (sym-cat ?ans)))
)

(defrule ask-feedback
    =>
    (printout t "Do you need feedback? (yes/no)" crlf)
    (bind ?ans (read))
    (assert (feedback (sym-cat ?ans)))
)