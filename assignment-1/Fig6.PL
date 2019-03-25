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

r4:-
    call(fact(c)), !,
    assert(fact(l)).

r5:-
    call(fact(l)), !,
    call(fact(m)), !,
    assert(n);
    r4.

run:-
    r5, r4, r3, r2, r1.