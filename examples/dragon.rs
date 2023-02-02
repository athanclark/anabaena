use turtle::{Turtle, Distance};
use anabaena::*;

#[derive(PartialEq, Eq, Debug, Hash, Clone)]
enum SierpinskiAlphabet {
    F,
    G,
    Left,
    Right,
}

fn main() {
    use SierpinskiAlphabet::*;

    let rules: LRulesHash<(), SierpinskiAlphabet> = LRulesHash::from([
        (
            F,
            LRulesQualified {
                no_context: Some(vec![
                    (1, Box::new(|_| vec![
                        F,
                        Left,
                        G,
                    ]))
                ]),
                ..LRulesQualified::default()
            }
        ),
        (
            G,
            LRulesQualified {
                no_context: Some(vec![
                    (1, Box::new(|_| vec![
                        F,
                        Right,
                        G,
                    ]))
                ]),
                ..LRulesQualified::default()
            }
        ),
    ]);

    let mut lsystem = LSystem {
        string: vec![F],
        rules,
        context: (),
        mk_context: Box::new(|_,_| ()),
    };


    let set = lsystem.nth(10).unwrap();


    let mut turtle = Turtle::new();
    turtle.use_degrees();
    turtle.set_speed(25);
    turtle.pen_up();
    turtle.go_to([100.0, 100.0]);
    turtle.pen_down();

    const UNIT_DISTANCE: Distance = 4.0;

    for x in set {
        match x {
            F => {
                turtle.forward(UNIT_DISTANCE);
            }
            G => {
                turtle.forward(UNIT_DISTANCE);
            }
            Left => {
                turtle.left(90.0);
            }
            Right => {
                turtle.right(90.0);
            }
        }
    }
}