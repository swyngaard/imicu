
#ifndef __KDOP_H__
#define __KDOP_H__

#include <vector>
#include "tools.h"

namespace pilar
{
	class CA
	{
	public:
		CA() {};
		
		struct Connection
		{
			int index;
			float nscal;
			float iscal;
			float jscal;
		};
		
		Vector3f pn;
		Vector3f iproj;
		Vector3f jproj;
		
		std::vector<Connection> array;
	};

	class KDOP
	{
	public:
		
		KDOP(int k = 14);
		KDOP(std::vector<Vector3f>& vertex, int k = 14);
		KDOP(const KDOP& kdop);
		~KDOP();
		
		int K;
		
		std::vector<Vector3f>& debug();
		std::vector<Vector3f>& debugVertices();
		
		void update(std::vector<Vector3f>& vertex);
		
		bool collides(const KDOP* kdop);
		bool merge(const KDOP* kdop);
		
		std::vector<Vector3i>& getNormals();
		bool** getDegenerateMatrix();
		float* getDistances();
		
		bool setDistances(const KDOP* kdop);
		
		void buildDegenerateMatrix();
		
	protected:
		
		std::vector<Vector3f> vertices;
		std::vector<Vector3i> normal;
		
		float *distance;
		bool **ndg;
		
		void initialise(int k);
		
		void build6(std::vector<Vector3f>& vertex);
		void build14(std::vector<Vector3f>& vertex);
		void build18(std::vector<Vector3f>& vertex);
		void build26(std::vector<Vector3f>& vertex);
		
		void addNormals6();
		void addNormals14();
		void addNormals18();
		void addNormals26();
		
		void setDegenerateMatrix();
	};
}

#endif
