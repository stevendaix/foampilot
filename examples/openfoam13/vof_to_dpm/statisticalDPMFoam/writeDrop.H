//Using standard c++ functions let's create the dictionary with all required variables  
std::ofstream file_1("d_post", std::ios::app);  
file_1 << d << ";" << std::endl;
file_1.close();

std::ofstream file_2("r_post", std::ios::app);  
file_2 << r << ";" << std::endl;
file_2.close();

std::ofstream file_3("vel_x_post", std::ios::app);  
file_3 << vel.component(vector::X) << ";" << std::endl;
file_3.close();

std::ofstream file_4("vel_y_post", std::ios::app);  
file_4 << vel.component(vector::Y) << ";" << std::endl;
file_4.close();

std::ofstream file_5("vel_zZ_post", std::ios::app);  
file_5 << vel.component(vector::Z) << ";" << std::endl;
file_5.close();


