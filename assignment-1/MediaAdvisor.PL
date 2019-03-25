:- dynamic environment/1.
:- dynamic job/1.
:- dynamic feedback/1.
:- dynamic situation/1.
:- dynamic response/1.
:- dynamic medium/1.

prompt(Environment, Job, Feedback, Medium):-
    assert(environment(Environment)),
    assert(job(Job)),
    assert(feedback(Feedback)),
    seek_medium(Medium).

seek_medium(Medium):-
    call(medium_roleplay), !,
	Medium = "roleplay";
    call(medium_videocassette), !,
    Medium = "videocassette";
    call(medium_lecture), !,
    Medium = "lecture";
    call(medium_workshop), !,
    Medium = "workshop";
    Medium = "None".

situation_verbal:-
    call(environment(papers)) ;
    call(environment(manuals)) ;
    call(environment(documents)) ;
    call(environment(textbooks)),
    assert(situation(verbal)).

situation_visual:-
    call(environment(pictures));
    call(environment(illustrations));
    call(environment(photographs));
    call(environment(diagrams)),
    assert(situation(visual)).

situation_physical:-
    call(environment(machines));
    call(environment(buildings));
    call(environment(tools)),
    assert(situation(physical)).

situation_symbolic:-
    call(environment(numbers));
    call(environment(formulas));
    call(environment(programs)),
    assert(situation(symbolic)).

response_oral:-
    call(job(lecturing)) ;
    call(job(formulas)) ;
    call(job(programs)),
    assert(response(oral)).

response_hands_on:-
    call(job(building)) ;
    call(job(repairing)) ;
    call(job(troubleshooting)),
    assert(response(hands-on)).

response_documented:-
    call(job(writing));
    call(job(typing));
    call(job(drawing)),
    assert(response(documented)).

response_analytical:-
    call(job(evaluating));
    call(job(reasoning));
    call(job(investigating)),
    assert(response(analytical)).

medium_workshop:-
    feedback(yes),
    call(situation(physical)),!,
    call(response(hands-on)),!,
    assert(medium(workshop));
    situation_physical,
    response_hands_on.

medium_lecture:-
    feedback(yes),
    call(situation(symbolic)), !,
    call(response(analytical)), !,
    assert(medium(lecture));
    situation_symbolic,
    response_analytical;
    call(situation(visual)), !,
    call(response(oral)), !,
    assert(medium(lecture));
    situation_visual,
    response_oral;
    call(situation(verbal)), !,
    call(response(analytica)), !,
    assert(medium(lecture)).

medium_videocassette:-
    feedback(yes),
    call(situation(visual)), !,
    call(response(documented)), !,
    assert(medium(videocassette));
    situation_visual,
    response_documented.

medium_roleplay:-
    feedback(yes),
    call(situation(verbal)), !,
    call(response(oral)), !,
    assert(medium(roleplay));
    situation_verbal,
    response_oral.