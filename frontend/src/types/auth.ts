export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  token: string;
  token_type: string;
  expires_in: number;
  user: UserProfile;
}

export interface UserProfile {
  id: number;
  username: string;
  full_name: string;
  role: "ADMIN" | "TEACHER";
  active: boolean;
  created_at: string;
  last_login_at: string | null;
}

export interface CreateUserRequest {
  username: string;
  password: string;
  full_name: string;
  role: string;
}

export interface UpdateUserRequest {
  full_name?: string;
  password?: string;
  role?: string;
}
