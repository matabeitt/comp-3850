:- dynamic fact/1.

fact(a).
fact(b).
fact(c).
fact(d).
fact(e).

r1:-
    call(fact(y)), !,
    call(fact(d)), !,
    assert(fact(z));
    r2.

r2:-
    call(fact(x)), !,
    fact(b),
    fact(e),
    assert(fact(y));
    r3.

r3:-
    call(fact(a)),
    assert(fact(x)).

run:-
    r1, r2, r3.