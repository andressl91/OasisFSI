
res1 = 0.025;
res2 = 0.005 ;
res3 = 0.0001875;

Point(1) = {0, 0, 0, res1};
Point(2) = {2.2, 0, 0, res1};
Point(3) = {2.2, 0.41, 0, res1};
Point(4) = {0, 0.41, 0, res1};

Point(5) = {0.20, 0.20, 0, res3};
Point(6) = {0.20, 0.25, 0, res3};
Point(7) = {0.20, 0.15, 0, res3};
Point(8) = {0.25, 0.20, 0, res3};
Point(9) = {0.15, 0.20, 0, res3};

Point(10) = {0.70, 0.25, 0, res2};
Point(11) = {0.70, 0.15, 0, res2};
Point(12) = {0.20, 0.275, 0, res2};
Point(13) = {0.20, 0.125, 0, res2};
Point(14) = {0.0, 0.275, 0, res1};
Point(15) = {0.0, 0.125, 0, res1};


Circle(10) = {8, 5, 6};
Circle(20) = {6, 5, 9};
Circle(30) = {9, 5, 7};
Circle(40) = {7, 5, 8};

Line(41) = {11, 10};
Line(42) = {10, 12};
Line(43) = {12, 14};
Line(44) = {11, 13};
Line(47) = {13, 15};
Line(48) = {15, 14};
Line(49) = {2, 3};
Line(50) = {3, 4};
Line(51) = {4, 14};
Line(52) = {15, 1};
Line(53) = {1, 2};
Line Loop(54) = {49, 50, 51, -43, -42, -41, 44, 47, 52, 53};
Plane Surface(55) = {54};
Line Loop(56) = {41, 42, 43, -48, -47, -44};
Line Loop(57) = {30, 40, 10, 20};
Plane Surface(58) = {56, 57};


Field[1] = BoundaryLayer;
Field[1].EdgesList = {10,20,30,40};
Field[1].hwall_n = 0.0001;
Field[1].ratio = 1.2;
Field[1].thickness = 0.0015;
Field[1].Quads = 0;
BoundaryLayer Field = 1;
