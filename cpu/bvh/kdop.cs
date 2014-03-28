        struct Table
        {
            public Vector3 PerpNormal;
            public Vector3 iProjector;
            public Vector3 jProjector;
            public Boolean NonDegenerate;
            public List<Possible> GoodIndices;
        }

        struct Possible
        {
            public Int32 K;
            public Single nScal;
            public Single iScal;
            public Single jScal;
        }

        static private Single Determinant(float[][] mat)
        {
            float Det00 = (mat[0][0] * mat[1][1] * mat[2][2]) - (mat[0][0] * mat[1][2] * mat[2][1]);
            float Det01 = (mat[0][1] * mat[1][2] * mat[2][0]) - (mat[0][1] * mat[1][0] * mat[2][2]);
            float Det02 = (mat[0][2] * mat[1][0] * mat[2][1]) - (mat[0][2] * mat[1][1] * mat[2][0]);
            return Det00 + Det01 + Det02;
        }

        static private Table[][] _table26;
        static private Table[][] Table26
        {
            get
            {
                if (_table26 == null)
                {
                    kDOP kDOP = new kDOP();
                    kDOP.K = 26;
                    PlaneN[] Planes = new PlaneN[26];
                    for (int i = 0; i < 26; i++)
                    {
                        Planes[i] = new PlaneN(kDOP.getPlaneNormal(i), 0);
                    }

                    float[][] mat = new float[3][];
                    for (int i = 0; i < 3; i++)
                    {
                        mat[i] = new float[3];
                    }

                    _table26 = new Table[Planes.Length][];
                    for (int i = 0; i < Planes.Length; i++)
                    {
                        _table26[i] = new Table[Planes.Length];
                        for (int j = i + 1; j < Planes.Length; j++)
                        {
                            Vector3 tmp = Planes[i].Normal + Planes[j].Normal;
                            tmp.X = Math.Abs(tmp.X);
                            tmp.Y = Math.Abs(tmp.Y);
                            tmp.Z = Math.Abs(tmp.Z);
                            if ((tmp.X != 0) && (tmp.Y != 0) && (tmp.Z != 0))
                            {
                                Boolean t = (tmp.X == tmp.Y);
                                if (t == true)
                                    _table26[i][j].NonDegenerate = !(tmp.X == tmp.Z);
                                else
                                    _table26[i][j].NonDegenerate = true;
                            }
                            else
                            {
                                if ((tmp.X != 0) && (tmp.Y != 0))
                                {
                                    _table26[i][j].NonDegenerate = !(tmp.X == tmp.Y);
                                }
                                else
                                {
                                    if ((tmp.X != 0) && (tmp.Z != 0))
                                    {
                                        _table26[i][j].NonDegenerate = !(tmp.X == tmp.Z);
                                    }
                                    else
                                    {
                                        if ((tmp.Y != 0) && (tmp.Z != 0))
                                        {
                                            _table26[i][j].NonDegenerate = !(tmp.Y == tmp.Z);
                                        }
                                        else
                                        {
                                            _table26[i][j].NonDegenerate = false;
                                        }
                                    }
                                }
                            }

                            _table26[i][j].PerpNormal = Vector3.Cross(Planes[i].Normal, Planes[j].Normal);
                            Vector3 v = Vector3.Cross(_table26[i][j].PerpNormal, Planes[j].Normal);
                            _table26[i][j].iProjector = v / Vector3.Dot(v, Planes[i].Normal);
                            v = Vector3.Cross(_table26[i][j].PerpNormal, Planes[i].Normal);
                            _table26[i][j].jProjector = v / Vector3.Dot(v, Planes[j].Normal); 
                            _table26[i][j].GoodIndices = new List<Possible>();

                            for (int k = 0; k < Planes.Length; k++)
                            {
                                if ((k != i) && (k != j))
                                {
                                    mat[0][0] = Planes[i].Normal.X;
                                    mat[0][1] = Planes[i].Normal.Y;
                                    mat[0][2] = Planes[i].Normal.Z;
                                    mat[1][0] = Planes[j].Normal.X;
                                    mat[1][1] = Planes[j].Normal.Y;
                                    mat[1][2] = Planes[j].Normal.Z;
                                    mat[2][0] = Planes[k].Normal.X;
                                    mat[2][1] = Planes[k].Normal.Y;
                                    mat[2][2] = Planes[k].Normal.Z;
                                    Single Determinan = Determinant(mat);
                                    if (Determinan != 0)
                                    {
                                        Boolean Good = true;
                                        for (int m = 0; m < Planes.Length; m++)
                                        {
                                            if ((m != i) && (m != j) && (m != k))
                                            {
                                                mat[0][0] = Planes[i].Normal.X;
                                                mat[0][1] = Planes[i].Normal.Y;
                                                mat[0][2] = Planes[i].Normal.Z;
                                                mat[1][0] = Planes[j].Normal.X;
                                                mat[1][1] = Planes[j].Normal.Y;
                                                mat[1][2] = Planes[j].Normal.Z;
                                                mat[2][0] = Planes[m].Normal.X;
                                                mat[2][1] = Planes[m].Normal.Y;
                                                mat[2][2] = Planes[m].Normal.Z;

                                                Single Det2 = Determinant(mat);

                                                mat[0][0] = Planes[i].Normal.X;
                                                mat[0][1] = Planes[i].Normal.Y;
                                                mat[0][2] = Planes[i].Normal.Z;
                                                mat[1][0] = Planes[m].Normal.X;
                                                mat[1][1] = Planes[m].Normal.Y;
                                                mat[1][2] = Planes[m].Normal.Z;
                                                mat[2][0] = Planes[k].Normal.X;
                                                mat[2][1] = Planes[k].Normal.Y;
                                                mat[2][2] = Planes[k].Normal.Z;

                                                Single Det3 = Determinant(mat);

                                                mat[0][0] = Planes[m].Normal.X;
                                                mat[0][1] = Planes[m].Normal.Y;
                                                mat[0][2] = Planes[m].Normal.Z;
                                                mat[1][0] = Planes[j].Normal.X;
                                                mat[1][1] = Planes[j].Normal.Y;
                                                mat[1][2] = Planes[j].Normal.Z;
                                                mat[2][0] = Planes[k].Normal.X;
                                                mat[2][1] = Planes[k].Normal.Y;
                                                mat[2][2] = Planes[k].Normal.Z;

                                                Single Det4 = Determinant(mat);

                                                if (Determinan > 0)
                                                {
                                                    if (!((Det2 < 0) || (Det3 < 0) || (Det4 < 0)))
                                                    {
                                                        Good = false;
                                                        break;
                                                    }
                                                }
                                                else
                                                {
                                                    if (!((Det2 > 0) || (Det3 > 0) || (Det4 > 0)))
                                                    {
                                                        Good = false;
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                        if (Good)
                                        {
                                            Possible ps = new Possible();
                                            ps.K = k;
                                            ps.iScal = Vector3.Dot(_table26[i][j].iProjector, Planes[k].Normal);
                                            ps.jScal = Vector3.Dot(_table26[i][j].jProjector, Planes[k].Normal);
                                            ps.nScal = Vector3.Dot(_table26[i][j].PerpNormal, Planes[k].Normal);
                                            _table26[i][j].GoodIndices.Add(ps);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                return _table26;
            }
        }

        static public void CreateMeshNew26(kDOP kDOP, out Vector3[] Vertices)
        {
#if DEBUG
            Int32 Id = SpeedChecker.StartCheck("Create Segments");
#endif
            Single[] M = new float[26];
            for (int i = 0; i < 13; i++)
            {
                M[i] = -kDOP.Min[i];
                M[i+13] = kDOP.Max[i];
            }

            List<Vector3> LineSegments = new List<Vector3>();
            Table[][] Tabella = Table26;
            for (int i = 0; i < 26; i++)
            {
                for (int j = 0; j < 26; j++)
                {
                    if (Tabella[i][j].NonDegenerate == false)
                    {
                        continue;
                    }
                    else
                    {
                        Int32 N = Tabella[i][j].GoodIndices.Count;
                        if (N > 0)
                        {
                            Single Min = Single.MinValue;
                            Single Max = Single.MaxValue;
                            for (int k = 0; k < N; k++)
                            {
                                Possible S = Tabella[i][j].GoodIndices[k];

                                if (S.nScal < 0)
                                {
                                    Single Value = (M[S.K] - M[i] * S.iScal - M[j] * S.jScal) / S.nScal;
                                    if (Value > Min)
                                        Min = Value;
                                }
                                if (S.nScal > 0)
                                {
                                    Single Value = (M[S.K] - M[i] * S.iScal - M[j] * S.jScal) / S.nScal;
                                    if (Value < Max)
                                        Max = Value;
                                }
                                if (Max < Min)
                                    break;
                            }
                            if (Min < Max)
                            {
                                Vector3 X = M[i] * Tabella[i][j].iProjector + M[j] * Tabella[i][j].jProjector;
                                LineSegments.Add(X + (Min * Tabella[i][j].PerpNormal));
                                LineSegments.Add(X + (Max * Tabella[i][j].PerpNormal));
                            }
                        }
                    }
                }
            }
            Vertices = LineSegments.ToArray();
#if DEBUG
            SpeedChecker.EndCheck(Id);
#endif
        }