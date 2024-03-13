from src.data.mesh_loss_function import mesh_loss_function
from src.features.generate_flame_mesh import generate_flame_mesh
from src.features.image_to_pointcloud_landmark import image_to_pointcloud_landmark
import numpy as np
from tqdm import tqdm
from src.features.generate_flame_mesh import generate_flame_mesh
import pygad
from src.visualization.visualize_mesh import visualize_mesh

class mesh_GA_wrapper:
    def __init__(self):
        self.ga_instance=None

        self.landmark=None
        self.fitness_functions=["landmarks","mesh"]
        self.reduce_landmarks=True
        pass

    def setRefereneImage(self, image, visualize=True):
        """
        Set the reference image for the object and update the landmark based on the input image.

        Args:
            self: The object itself.
            image: The reference image to be set.

        Returns:
            None
        """
        self.landmark = image_to_pointcloud_landmark(image, plot_image=visualize, plot_pointcloud=False)
        pass

    def setFitnessFunctions(self, fitness_functions):
        """
        Set the fitness functions for the object.

        Args:
            fitness_functions (list): A list of strings representing the names of the fitness functions.

        Raises:
            ValueError: If fitness_functions is not a list, or if it is an empty list.

        Returns:
            None
        """
        # Check if fitness_functions is a list
        if type(fitness_functions) is not list:
            raise ValueError("fitness_functions invalid. Please set the fitness_functions as a list.")

        # Check if fitness_functions is not an empty list
        if len(fitness_functions) <= 0:
            raise ValueError("fitness_functions invalid. Please set the fitness_functions with at least one fitness function.")

        # Convert fitness_functions to lowercase
        fitness_functions = [x.lower() for x in fitness_functions]

        # Set the fitness_functions attribute
        self.fitness_functions = fitness_functions

    def setGeneticAlgorithm(self, ga_parameters):
        self.ga_instance=pygad.GA(**ga_parameters,fitness_func=self.__meshFitnessFunction,on_generation=self.__on_generation,suppress_warnings=True)
        pass

    def runGA(self,plot_fitness=False):
        self.progress_bar = tqdm(total=self.ga_instance.num_generations)

        self.ga_instance.run()

        if plot_fitness:
            self.ga_instance.plot_fitness()

        self.solution, self.solution_fitness,_ = self.ga_instance.best_solution()

        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=self.solution_fitness))
        pass

    def getMesh(self,getParams=False,visualize=False):
        solution=self.solution

        sol_shape_params=np.array([solution[:100]])
        sol_pose_params=np.array([solution[100:106]])
        sol_expression_params=np.array([solution[106:]])

        if visualize or not getParams:
            Mesh,Flame_landmarks=generate_flame_mesh(sol_shape_params,sol_pose_params,sol_expression_params,visualize=False)

        if visualize:
            mesh_loss_function(Mesh,Flame_landmarks,self.landmark,visualize=True,apply_loss=self.fitness_functions,reduce_landmarks=True)

        if getParams:
            return {"shape_params":sol_shape_params,"pose_params":sol_pose_params,"expression_params":sol_expression_params}
        else:
            return (Mesh,Flame_landmarks)

    def __meshFitnessFunction(self, ga_instance, x, solution_idx):
        """
        Calculate the fitness of a given mesh based on the provided parameters and fitness functions.

        Args:
            ga_instance (GeneticAlgorithm): The genetic algorithm instance.
            x (numpy.ndarray): The parameters of the mesh.
            solution_idx (int): The index of the solution.

        Raises:
            ValueError: If the reference image is not defined.

        Returns:
            list: The total loss of the mesh.
        """
        # Check if reference image is defined
        if self.landmark is None:
            raise ValueError("Reference image not defined. "
                             "Please set the reference image using the setReferenceImage() method.")

        # Extract parameters
        temp_shape_params = np.array([x[:100]])
        temp_pose_params = np.array([x[100:106]])
        temp_expression_params = np.array([x[106:]])

        # Generate mesh and landmarks
        Mesh, Flame_landmarks = generate_flame_mesh(temp_shape_params,
                                                   temp_pose_params,
                                                   temp_expression_params)

        # Calculate loss
        loss = mesh_loss_function(Mesh,
                                  Flame_landmarks,
                                  self.landmark,
                                  apply_loss=self.fitness_functions,
                                  reduce_landmarks=self.reduce_landmarks)

        # Calculate total loss
        total_loss = [-1 * loss[i] for i in loss]

        return total_loss

    def __on_generation(self, ga_instance):
        """
        Update the progress bar and close it after the last generation.

        Args:
            ga_instance (GeneticAlgorithm): The genetic algorithm instance.
        """
        # Get the total number of generations and the current generation number
        total_generations = ga_instance.num_generations
        current_generation = ga_instance.generations_completed

        # Update the progress bar for tqdm
        self.progress_bar.update()

        # Close the progress bar after the last generation
        if current_generation == total_generations:
            self.progress_bar.close()

    