/** Whether the server refused a request for want of a key, so the UI must sign in before it can go on. */
class AuthStore {
  required = $state(false);
}

export const authStore = new AuthStore();
