//Author: Adriel Kim
// 7-6-2020
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cstring>
using namespace std;
//How much precision do I need to preserve?

int main(){
    string line;
    ifstream str;
    string name = "img1.txt";
    str.open("./FitTextFiles/"+name);
    if(str.is_open()){
        
        getline(str,line);
    }
    int count = 10;
    if(str.is_open()){
        while(getline(str,line) && count > 0){

            char cstr[line.size()+1];
            strcpy(cstr, line.c_str());
            double num = atof(cstr);
            cout<<num<<'\n';
            count--;
        }
        str.close();
    }
    else{
        cout<<"Unable to open file"<<endl;
    }
    return 0;
}
