#include <torch/script.h>
#include <unistd.h>
#include <dirent.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#define ROUND 50

void getFiles(std::string path, std::vector<std::string>& files) {
    DIR *dir;
    struct dirent *ptr;
    char base[1000];

    if ((dir=opendir(path.c_str())) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
            continue;
        else if(ptr->d_type == 8)    ///file
            //printf("d_name:%s/%s\n",basePath,ptr->d_name);
            files.push_back(ptr->d_name);
        else if(ptr->d_type == 10)    ///link file
            //printf("d_name:%s/%s\n",basePath,ptr->d_name);
            continue;
        else if(ptr->d_type == 4)    ///dir
        {
            files.push_back(ptr->d_name);
            /*
               memset(base,'\0',sizeof(base));
               strcpy(base,basePath);
               strcat(base,"/");
               strcat(base,ptr->d_nSame);
               readFileList(base);
               */
        }
    }
    closedir(dir);
}

int main() {
    std::vector<std::string> modules;
    getFiles("modules", modules);

    for (auto module_name:modules) {
        std::string name = "./modules/" + module_name;
        std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(name.c_str());

        assert(module != nullptr);
        module->to(torch::kCUDA);
        torch::set_num_threads(1);

        double delta;
        double delta_sec;

        int batch[] = {1, 2, 4, 8, 16};
        //int batch[] = {1, 1};
        //int batch[] = {1, 16, 8, 4, 2, 1};
        int size = (module_name != "inception_v3") ? 224 : 299;

        for (int i = 0; i < sizeof(batch)/sizeof(batch[0]); i++) {
            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(torch::zeros({ batch[i], 3, size, size }).to(torch::kCUDA));
            {
                auto output_init = module->forward(inputs);
                auto embed_init = output_init.toTensor();
                embed_init.to(at::kCPU);
            }
            auto start = std::chrono::system_clock::now(); // do something...
            for (int j = 0; j < ROUND; j++) {
                auto output = module->forward(inputs);
                if (j == ROUND - 1) {
                    auto embed = output.toTensor();
                    embed.to(at::kCPU);
                }
            }

            auto end = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            delta = double(duration.count());

            delta_sec = delta * 1000 * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
            printf("Name:%10s Batch:%5d Eval:0ms Trace:%8.3fms Eval Cuda Size:0KB Traced Cuda Size:0KB\n", module_name.c_str(), batch[i], delta_sec/ROUND);

        }
    }
    return 0;

}
