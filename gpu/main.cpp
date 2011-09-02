
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>

#include "hair.h"
#include "constants.h"

pilar::Hair* hair = NULL;

void writeToFile(int count)
{
	//Convert the file number to a string
	std::string number;
	std::stringstream sout;
	sout << std::setfill('0') << std::setw(4) << count;
	
	//Construct the output file name
	std::string folder("output");
	mkdir(folder.c_str(), 0777);
	std::string prefix = std::string("frame") + std::string(sout.str());
	std::string file = folder + std::string("/") + prefix + std::string(".lxs");
	std::ofstream fout(file.c_str());
	
	//Output the name of the filemake
	fout << "LookAt 0 0 -35 0 0 0 0 1 0" << std::endl;
	fout << "Film \"fleximage\"" << std::endl;
	fout << "\t\t\"string filename\" [\"" << prefix << "\"]" << std::endl;
	
	//Open new file and write out scene template
	std::ifstream fin("template.lxs");
	char str[2048];
	
	while(!fin.eof())
	{
		fin.getline(str, 2048);
		fout << str << std::endl;
	}
	
	//Close LuxRender scene template
	fin.close();
	
	//Write out individual follicle positions
	for(int i = 0; i < hair->numStrands; i++)
	{
		for(int j = 0; j < hair->strand[i]->numParticles; j++)
		{
			pilar::Particle* particle = hair->strand[i]->particle[j];
			
			fout << "AttributeBegin" << std::endl;
			fout << "Material \"matte\" \"color Kd\" [0.1 0.1 0.8 ]" << std::endl;
			fout << "\tTranslate " << particle->position.x << " " << particle->position.y << " " << particle->position.z << std::endl;
			fout << "\tShape \"sphere\" \"float radius\" 0.25" << std::endl;
			fout << "AttributeEnd" << std::endl;
			fout << std::endl;
		}
	}
	
	fout << "WorldEnd" << std::endl;
	
	//Close the output file
	fout.close();
}


int main()
{
	std::cout << "Hair Simulation" << std::endl;	
	
	float elapsed = 0.0f;
	float dt = 1.0f/50.0f; //50 Frames per second
	float total = 1.0f * dt; //total time of the simulation
	int fileCount = 0;
	
	pilar::Vector3f root;
	
	std::vector<pilar::Vector3f> roots;
	
	roots.push_back(root);
	std::cout << "Strands: " << roots.size() << std::endl;
	
	hair = new pilar::Hair(roots.size(), NUMPARTICLES, MASS, K_EDGE, K_BEND, K_TWIST, K_EXTRA, D_EDGE, D_BEND, D_TWIST, D_EXTRA, LENGTH, roots);
	
	while(elapsed < total)
	{
		hair->update(dt);
		
//		writeToFile(fileCount);
		
		elapsed += dt;
		fileCount++;
	}
	
	hair->release();
	
	delete hair;
	
	hair = NULL;
	
	return 0;
}

