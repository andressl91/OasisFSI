res1 = 0.008;
res2 = 0.001;
// Tube domain
Point(1) = {0, 0, 0, 0.1};
Point(2) = {0, 0.41, 0, 0.1};
Point(3) = {2.5, 0.41, 0, 0.1};
Point(4) = {2.5, 0, 0, 0.1};

// Circle
Point(5) = {0.2, 0.2, 0, 1};
Point(7) = {0.2, 0.15, 0, res2};
Point(8) = {0.2, 0.25, 0, res2};
Point(9) = {0.15, 0.2, 0, res2};
Point(10) = {0.2489897949, 0.21, 0.0, res2};
Point(11) = {0.2489897949, 0.19, 0.0, res2};

//Flag end points
Point(12) = {0.6, 0.19, 0, res1};
Point(13) = {0.6, 0.21, 0, res1};

Line(14) = {1, 2};
Line(15) = {2, 3};
Line(16) = {3, 4};
Line(17) = {4, 1};

Circle(18) = {9, 5, 8};
Circle(19) = {8, 5, 10};
Circle(20) = {10, 5, 11};
Circle(21) = {11, 5, 7};
Circle(22) = {7, 5, 9};

Line(24) = {13, 12};
Point(61) = {0.3, 0.21, 0, res1};
Point(62) = {0.3, 0.19, 0, res1};

Line(63) = {61,62};
Line(64) = {61,10};
Line(65) = {62,11};

Line(66) = {61,13};
Line(67) = {62,12};


// Fluid surface
Line loop(26) = {14,15,16,17};
Line loop(27) = {18,19,-64,66,24,-67,65,21,22};
Plane Surface(102) = {26, 27};

Field[1] = BoundaryLayer;
Field[1].EdgesList = {18,19,20,21,22};
Field[1].hwall_n = 0.0005;
Field[1].ratio = 1.2;
Field[1].thickness = 0.002;
Field[1].Quads = 0;
BoundaryLayer Field = 1;

// Solid Surface
Line loop(28) = {-20,-64,63,65};
Plane Surface(103) = {28};
Line loop(29) = {-63,66,24,-67};
Plane Surface(104) = {29};

Transfinite Line{-20,-64,63,65} = 10;
Transfinite Surface{103};
Transfinite Line{-63,66,24,-67} = 10;
Transfinite Surface{104};
