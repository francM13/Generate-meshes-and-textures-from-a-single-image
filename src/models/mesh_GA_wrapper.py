from src.data.mesh_loss_function import mesh_loss_function
from src.features.generate_flame_mesh import generate_flame_mesh
from src.features.image_to_pointcloud_landmark import image_to_pointcloud_landmark
import numpy as np
from tqdm import tqdm
from src.features.generate_flame_mesh import generate_flame_mesh
import pygad
from src.visualization.visualize_mesh import visualize_mesh

class mesh_GA_wrapper:
    def __init__(self, quality="medium"):
        """
        Initialize the mesh_GA_wrapper object.

        Args:
            quality (str or int): The quality of the genetic algorithm. It can be a string
                ("high", "medium", or "low") or an integer representing the number of generations.
                Default is "high".
        """
        # Initialize the genetic algorithm parameters
        if isinstance(quality, int):
            num_generations = quality
        elif quality == "high":
            num_generations = 200
        elif quality == "medium":
            num_generations = 100
        else:
            num_generations = 50

        # Set the genetic algorithm parameters
        ga_params = {
            "num_generations": num_generations,  # Number of generations
            "num_parents_mating": 4,  # Number of parents in mating
            "sol_per_pop": 16,  # Number of solutions per population
            "parent_selection_type": "sss",  # Parent selection type
            "crossover_type": "single_point",  # Crossover type
            "mutation_type": "random",  # Mutation type
            "mutation_percent_genes": 10,  # Percentage of genes mutated
            "keep_parents": 2,  # Number of parents to keep
            "keep_elitism": 1,  # Number of elitist solutions to keep
            "gene_space": {'low': -2, 'high': 2},  # Gene space
            "num_genes": 156,  # Number of genes
        }
        self.setGeneticAlgorithm(ga_params)  # Set the genetic algorithm

        self.landmark = None  # Initialize the landmark
        self.fitness_functions = ["landmarks", "mesh"]  # Initialize the fitness functions
        self.reduce_landmarks = True  # Flag to reduce landmarks

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
        """
        Set the genetic algorithm instance for the object.

        Args:
            ga_parameters (dict): The parameters for the genetic algorithm.

        Returns:
            None
        """
        # Set the genetic algorithm instance
        self.ga_instance = pygad.GA(
            **ga_parameters,
            fitness_func=self.__meshFitnessFunction,
            on_generation=self.__on_generation,
            suppress_warnings=True
        )
        pass

    def run(self, plot_fitness=False):
        """
        Run the genetic algorithm and get the best solution.

        Args:
            plot_fitness (bool): Whether to plot the fitness values or not. Default is False.

        Returns:
            float: The fitness value of the best solution.
        """
        # Initialize the progress bar
        self.progress_bar = tqdm(total=self.ga_instance.num_generations)

        # Run the genetic algorithm
        self.ga_instance.run()

        # Plot the fitness values if required
        if plot_fitness:
            self.ga_instance.plot_fitness()

        # Get the best solution, its fitness value and other information
        self.solution, self.solution_fitness, _ = self.ga_instance.best_solution()

        # Print the fitness value of the best solution
        print(f"Fitness value of the best solution = {self.solution_fitness}")
        
        # Return the fitness value of the best solution
        return self.solution_fitness

    def getMesh(self, getParams=False, visualize=False):
        """
        Get the mesh corresponding to the solution of the genetic algorithm.

        Args:
            getParams (bool): If True, return the shape, pose and expression parameters.
            visualize (bool): If True, visualize the mesh.

        Returns:
            dict or tuple: If getParams is True, return a dictionary with the shape, pose and expression parameters. Otherwise, return a tuple with the mesh and the flame landmarks.
        """
        # Extract the solution from the wrapper
        solution = self.solution

        # Extract the shape, pose and expression parameters from the solution
        sol_shape_params = np.array([solution[:100]])
        sol_pose_params = np.array([solution[100:106]])
        sol_expression_params = np.array([solution[106:]])

        # Generate the mesh and flame landmarks
        if visualize or not getParams:
            Mesh, Flame_landmarks = generate_flame_mesh(
                sol_shape_params, sol_pose_params, sol_expression_params, visualize=False
            )

        # Visualize the mesh if requested
        if visualize:
            mesh_loss_function(
                Mesh,
                Flame_landmarks,
                self.landmark,
                visualize=True,
                apply_loss=self.fitness_functions,
                reduce_landmarks=True,
            )

        # Return the parameters or the mesh and flame landmarks
        if getParams:
            return {
                "shape_params": sol_shape_params,
                "pose_params": sol_pose_params,
                "expression_params": sol_expression_params,
            }
        else:
            return (Mesh, Flame_landmarks)

    def getNeutralMesh(self, getParams=False, visualize=False):
        """
        Get the neutral mesh corresponding to the solution of the genetic algorithm.

        Args:
            getParams (bool): If True, return the shape, pose and expression parameters.
            visualize (bool): If True, visualize the mesh.

        Returns:
            dict or tuple: If getParams is True, return a dictionary with the shape, pose and expression parameters. Otherwise, return a tuple with the mesh and the flame landmarks.
        """
        # Extract the neutral parameters from the solution
        solution = self.solution
        sol_shape_params = np.array([solution[:100]])  # Extract shape parameters
        sol_pose_params = np.array([[1, 0, 1, 0, 0, 0]])  # Extract pose parameters
        sol_expression_params = np.zeros([1, 50])  # Extract expression parameters

        # Generate the neutral mesh
        Mesh, Flame_landmarks = generate_flame_mesh(
            sol_shape_params, sol_pose_params, sol_expression_params, visualize=False
        )

        # Visualize the neutral mesh if requested
        if visualize:
            visualize_mesh(Mesh)

        # Return the parameters or the mesh and flame landmarks
        if getParams:
            return {
                "shape_params": sol_shape_params,
                "pose_params": sol_pose_params,
                "expression_params": sol_expression_params,
            }
        else:
            return (Mesh, Flame_landmarks)

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

    