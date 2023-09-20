# Pixel-Navigator
Zero-shot object navigation is a challenging task for home-assistance robots. This task emphasizes visual grounding, commonsense inference and locomotion abilities, where the first two are inherent in foundation models. But for the locomotion part, most works still depend on map-based planning approaches. The gap between RGB space and map space makes it difficult to directly transfer the knowledge from foundation models to navigation tasks. In this work, we propose a Pixel-guided Navigation skill (PixNav), which bridges the gap between the foundation models and the embodied navigation task. It is straightforward for recent foundation models to indicate an object by pixels, and with pixels as the goal specification, our method becomes a versatile navigation policy towards all different kinds of objects. Besides, our PixNav is a pure RGB-based policy that can reduce the cost of home-assistance robots. Experiments demonstrate the robustness of the PixNav which achieves 80+% success rate in the local path-planning task. To perform long-horizon object navigation, we design an LLM-based planner to utilize the commonsense knowledge between objects and rooms to select the best waypoint. Evaluations across both photorealistic indoor simulators and real- world environments validate the effectiveness of our proposed navigation strategy. 

<img width="806" alt="image" src="https://github.com/wzcai99/Pixel-Navigator/assets/115710611/3044748d-e9fc-493e-81a7-e9ae34a08259">

> [**Bridging Zero-Shot Object Navigation and Foundation Models through Pixel-Guided Navigation Skill**](https://arxiv.org/abs/2309.10309). 
> Wenzhe Cai, Siyuan Huang, Guangran Cheng, Yuxing Long, Peng Gao, Changyin Sun, Hao Dong. 

Codes and Models will be released soon.

## Example Trajectory in HM3D ##
![Find the chair](./assets/demo_chair.gif)
![Find the toilet](./assets/demo_toilet.gif)
