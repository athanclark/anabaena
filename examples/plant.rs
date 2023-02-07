use anabaena::*;
use std::collections::HashMap;
use turtle::{Angle, Distance, Point, Turtle};

#[derive(PartialEq, Eq, Debug, Hash, Clone)]
enum PlantAlphabet {
    X,
    F,
    Left,
    Right,
    Push,
    Pop,
}

fn main() {
    use PlantAlphabet::*;

    let rules: LRulesHash<(), _, _> = Box::new(|_| {
        HashMap::from([
            (
                X,
                LRulesQualified {
                    no_context: Some(LRulesSet::new(vec![(
                        1,
                        vec![
                            F, Left, Push, Push, X, Pop, Right, X, Pop, Right, F, Push, Right, F,
                            X, Pop, Left, X,
                        ],
                    )])),
                    ..LRulesQualified::default()
                },
            ),
            (
                F,
                LRulesQualified {
                    no_context: Some(LRulesSet::new(vec![(1, vec![F, F])])),
                    ..LRulesQualified::default()
                },
            ),
        ])
    });

    let mut lsystem = LSystem {
        string: vec![X],
        rules,
        context: (),
        mut_context: Box::new(|_, _| {}),
    };

    let set = lsystem.nth(5).unwrap();

    let mut turtle = Turtle::new();
    turtle.use_degrees();
    turtle.set_speed(25);
    turtle.pen_up();
    turtle.go_to([0.0, -200.0]);

    const UNIT_DISTANCE: Distance = 2.0;
    let mut lifo: Vec<(Point, Angle)> = vec![];
    let mut current_angle: Angle = 0.0;

    for x in set {
        match x {
            X => {}
            F => {
                turtle.pen_down();
                turtle.forward(UNIT_DISTANCE);
                turtle.pen_up();
            }
            Push => {
                lifo.push((turtle.position(), current_angle));
            }
            Pop => {
                let (pos, new_angle) = lifo.pop().unwrap();
                let diff = new_angle - current_angle;
                current_angle = new_angle;
                // restore the saved angle and position
                turtle.go_to(pos);
                turtle.right(diff);
            }
            Left => {
                // turn
                current_angle -= 25.0;
                turtle.left(25.0);
            }
            Right => {
                // turn
                current_angle += 25.0;
                turtle.right(25.0);
            }
        }
    }
}
